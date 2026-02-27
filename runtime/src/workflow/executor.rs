//! Executor — async tree-walking interpreter for workflow expressions.
//!
//! Evaluates an `Expr` tree recursively. Tokio provides the concurrency
//! scheduling (Fork → `FuturesUnordered`, Pipe → sequential, etc.).

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use futures::future::join_all;
use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::Value;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::context;
use crate::process::{self, ProcessId};
use crate::program::{self, ProgramName};

use super::WorkflowId;
use super::expr::Expr;

// =============================================================================
// Executor
// =============================================================================

/// Tree-walking interpreter for a single workflow.
///
/// Owns the cancellation token and urgency counters. All `eval` methods
/// take `&self` — interior mutability via atomics and Mutex.
pub(super) struct Executor {
    pub id: WorkflowId,
    username: String,
    pub cancel: CancellationToken,
    total: AtomicUsize,
    completed: AtomicUsize,
    running: Mutex<HashSet<ProcessId>>,
}

impl Executor {
    pub fn new(id: WorkflowId, username: String) -> Self {
        Executor {
            id,
            username,
            cancel: CancellationToken::new(),
            total: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
            running: Mutex::new(HashSet::new()),
        }
    }

    pub fn total(&self) -> usize {
        self.total.load(Ordering::Relaxed)
    }

    pub fn completed(&self) -> usize {
        self.completed.load(Ordering::Relaxed)
    }

    /// Top-level entry point.
    pub async fn run(&self, expr: &Expr) -> Result<Value, String> {
        self.eval(expr, None, self.cancel.clone()).await
    }

    // =========================================================================
    // Core eval
    // =========================================================================

    /// Recursive async tree-walk. Returns a boxed future to support recursion.
    fn eval<'a>(
        &'a self,
        expr: &'a Expr,
        input: Option<Value>,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send + 'a>> {
        Box::pin(async move {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            match expr {
                Expr::Literal { value } => Ok(value.clone()),
                Expr::Process { program_name, config } => {
                    self.eval_process(program_name, config.as_ref(), input, cancel).await
                }
                Expr::Pipe { stages } => self.eval_pipe(stages, input, cancel).await,
                Expr::Fork { branches } => self.eval_fork(branches, input, None, cancel).await,
                Expr::Map { function, over } => {
                    self.eval_map(function, over, input, None, cancel).await
                }
                Expr::Fold { function, init, over } => {
                    self.eval_fold(function, init, over, input, cancel).await
                }
                Expr::Cond { predicate, then, otherwise } => {
                    self.eval_cond(predicate, then, otherwise, input, cancel).await
                }
                Expr::Iterate { body, until } => {
                    self.eval_iterate(body, until, input, cancel).await
                }
                Expr::Take { k } => {
                    // Standalone Take outside a Pipe: truncate input array.
                    match input {
                        Some(Value::Array(mut arr)) => {
                            arr.truncate(*k);
                            Ok(Value::Array(arr))
                        }
                        Some(v) => Ok(v),
                        None => Ok(Value::Array(Vec::new())),
                    }
                }
            }
        })
    }

    // =========================================================================
    // Pipe
    // =========================================================================

    async fn eval_pipe(
        &self,
        stages: &[Expr],
        input: Option<Value>,
        cancel: CancellationToken,
    ) -> Result<Value, String> {
        let mut val = input;
        let mut i = 0;
        while i < stages.len() {
            // Peek ahead for Take
            let take_k = match stages.get(i + 1) {
                Some(Expr::Take { k }) => Some(*k),
                _ => None,
            };
            val = Some(match (&stages[i], take_k) {
                (Expr::Fork { branches }, Some(k)) => {
                    self.eval_fork(branches, val, Some(k), cancel.clone()).await?
                }
                (Expr::Map { function, over }, Some(k)) => {
                    self.eval_map(function, over, val, Some(k), cancel.clone()).await?
                }
                (stage, _) => self.eval(stage, val, cancel.clone()).await?,
            });
            i += if take_k.is_some() { 2 } else { 1 };
        }
        val.ok_or_else(|| "empty pipe".into())
    }

    // =========================================================================
    // Process
    // =========================================================================

    async fn eval_process(
        &self,
        program_name: &str,
        config: Option<&Value>,
        input: Option<Value>,
        cancel: CancellationToken,
    ) -> Result<Value, String> {
        self.total.fetch_add(1, Ordering::Relaxed);

        let name = ProgramName::parse(program_name)
            .map_err(|e| format!("Invalid program name '{program_name}': {e}"))?;

        program::install(&name).await
            .map_err(|e| format!("Failed to install '{program_name}': {e}"))?;

        let input_str = build_process_input(input.as_ref(), config);
        let (result_tx, result_rx) = oneshot::channel();

        let pid = process::spawn(
            self.username.clone(),
            name,
            input_str,
            None,
            true,
            Some(result_tx),
            Some(self.id),
        ).map_err(|e| e.to_string())?;

        self.running.lock().unwrap().insert(pid);
        self.push_weights();

        let result = tokio::select! {
            r = result_rx => r.unwrap_or(Err("process channel dropped".into())),
            _ = cancel.cancelled() => {
                process::terminate(pid, Err("cancelled".into()));
                Err("cancelled".into())
            }
        };

        self.running.lock().unwrap().remove(&pid);
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.push_weights();

        match result {
            Ok(output) => Ok(parse_output(&output)),
            Err(e) => Err(e),
        }
    }

    // =========================================================================
    // Fork
    // =========================================================================

    async fn eval_fork(
        &self,
        branches: &[Expr],
        input: Option<Value>,
        take_k: Option<usize>,
        cancel: CancellationToken,
    ) -> Result<Value, String> {
        if branches.is_empty() {
            return Ok(Value::Array(Vec::new()));
        }

        match take_k {
            // No Take: join_all preserves input order.
            None => {
                let futs: Vec<_> = branches.iter()
                    .map(|b| self.eval(b, input.clone(), cancel.clone()))
                    .collect();
                let results: Vec<Result<Value, String>> = join_all(futs).await;
                let values: Result<Vec<Value>, String> = results.into_iter().collect();
                Ok(Value::Array(values?))
            }
            // Take: FuturesUnordered for first-k-wins, cancel remaining.
            Some(k) => {
                let fork_cancel = cancel.child_token();
                let branch_cancels: Vec<_> = (0..branches.len())
                    .map(|_| fork_cancel.child_token())
                    .collect();
                let futs: Vec<_> = branches.iter()
                    .zip(branch_cancels.iter())
                    .map(|(b, bc)| self.eval(b, input.clone(), bc.clone()))
                    .collect();

                let mut stream = FuturesUnordered::from_iter(futs);
                let mut results = Vec::with_capacity(k);

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(val) => {
                            results.push(val);
                            if results.len() >= k {
                                fork_cancel.cancel();
                                break;
                            }
                        }
                        Err(e) if is_cancel_error(&e) => continue,
                        Err(e) => {
                            fork_cancel.cancel();
                            while stream.next().await.is_some() {}
                            return Err(e);
                        }
                    }
                }
                // Drain remaining cancelled futures.
                while stream.next().await.is_some() {}

                Ok(Value::Array(results))
            }
        }
    }

    // =========================================================================
    // Map
    // =========================================================================

    async fn eval_map(
        &self,
        function: &Expr,
        over: &Expr,
        input: Option<Value>,
        take_k: Option<usize>,
        cancel: CancellationToken,
    ) -> Result<Value, String> {
        let over_val = self.eval(over, input.clone(), cancel.clone()).await?;
        let items = match over_val {
            Value::Array(arr) => arr,
            other => vec![other],
        };

        if items.is_empty() {
            return Ok(Value::Array(Vec::new()));
        }

        match take_k {
            // No Take: join_all preserves input order.
            None => {
                let futs: Vec<_> = items.into_iter()
                    .map(|item| self.eval(function, Some(item), cancel.clone()))
                    .collect();
                let results: Vec<Result<Value, String>> = join_all(futs).await;
                let values: Result<Vec<Value>, String> = results.into_iter().collect();
                Ok(Value::Array(values?))
            }
            // Take: FuturesUnordered for first-k-wins, cancel remaining.
            Some(k) => {
                let map_cancel = cancel.child_token();
                let branch_cancels: Vec<_> = (0..items.len())
                    .map(|_| map_cancel.child_token())
                    .collect();
                let futs: Vec<_> = items.into_iter()
                    .zip(branch_cancels.iter())
                    .map(|(item, bc)| self.eval(function, Some(item), bc.clone()))
                    .collect();

                let mut stream = FuturesUnordered::from_iter(futs);
                let mut results = Vec::with_capacity(k);

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(val) => {
                            results.push(val);
                            if results.len() >= k {
                                map_cancel.cancel();
                                break;
                            }
                        }
                        Err(e) if is_cancel_error(&e) => continue,
                        Err(e) => {
                            map_cancel.cancel();
                            while stream.next().await.is_some() {}
                            return Err(e);
                        }
                    }
                }
                // Drain remaining cancelled futures.
                while stream.next().await.is_some() {}

                Ok(Value::Array(results))
            }
        }
    }

    // =========================================================================
    // Fold
    // =========================================================================

    async fn eval_fold(
        &self,
        function: &Expr,
        init: &Expr,
        over: &Expr,
        input: Option<Value>,
        cancel: CancellationToken,
    ) -> Result<Value, String> {
        let over_val = self.eval(over, input.clone(), cancel.clone()).await?;
        let items = match over_val {
            Value::Array(arr) => arr,
            other => vec![other],
        };

        let mut acc = self.eval(init, input, cancel.clone()).await?;

        for item in items {
            let fn_input = merge_sources(item, acc);
            acc = self.eval(function, Some(fn_input), cancel.clone()).await?;
        }

        Ok(acc)
    }

    // =========================================================================
    // Cond
    // =========================================================================

    async fn eval_cond(
        &self,
        predicate: &Expr,
        then: &Expr,
        otherwise: &Expr,
        input: Option<Value>,
        cancel: CancellationToken,
    ) -> Result<Value, String> {
        let pred_val = self.eval(predicate, input.clone(), cancel.clone()).await?;
        if is_truthy_value(&pred_val) {
            self.eval(then, input, cancel).await
        } else {
            self.eval(otherwise, input, cancel).await
        }
    }

    // =========================================================================
    // Iterate
    // =========================================================================

    async fn eval_iterate(
        &self,
        body: &Expr,
        until: &Expr,
        input: Option<Value>,
        cancel: CancellationToken,
    ) -> Result<Value, String> {
        const MAX_ITERATIONS: usize = 1000;

        let mut val = input;
        for i in 0..MAX_ITERATIONS {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }

            val = Some(self.eval(body, val, cancel.clone()).await?);
            let check = self.eval(until, val.clone(), cancel.clone()).await?;

            if is_truthy_value(&check) {
                return val.ok_or_else(|| "iterate: empty body output".into());
            }

            if i == MAX_ITERATIONS - 1 {
                return Err(format!("Loop exceeded {MAX_ITERATIONS} iterations"));
            }
        }

        val.ok_or_else(|| "iterate: no iterations".into())
    }

    // =========================================================================
    // Urgency
    // =========================================================================

    /// Push SRPT-based weight to all model arbiters.
    fn push_weights(&self) {
        let total = self.total.load(Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Relaxed);
        let remaining = total.saturating_sub(completed);

        let weight = if remaining > 0 {
            (total as f64 / remaining as f64).max(1.0)
        } else {
            1.0
        };

        let pid_values: HashMap<ProcessId, f64> = self.running
            .lock()
            .unwrap()
            .iter()
            .map(|&pid| (pid, 1.0))
            .collect();

        context::set_dag_weights(weight, pid_values);
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Parse a process output string into a Value.
fn parse_output(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_string()))
}

/// Convert a Value to String. Strings are unwrapped; others JSON-serialized.
pub(super) fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Truthy check on a Value.
fn is_truthy_value(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map_or(false, |f| f != 0.0),
        Value::String(s) => !s.is_empty() && s != "false" && s != "0" && s != "null",
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Check if an error is from cancellation.
fn is_cancel_error(e: &str) -> bool {
    e == "cancelled" || e == "cancelled by take"
}

/// Merge an item (primary) and accumulator (secondary) for fold.
/// Object values merge at top level; non-objects go into `_input`.
fn merge_sources(item: Value, acc: Value) -> Value {
    let mut obj = serde_json::Map::new();

    match item {
        Value::Object(map) => obj.extend(map),
        other => { obj.insert("_input".into(), other); }
    }
    match acc {
        Value::Object(map) => obj.extend(map),
        other => { obj.insert("_input".into(), other); }
    }

    Value::Object(obj)
}

/// Build the JSON input string for a process from its pipeline input
/// and static config.
///
/// - Object inputs are merged at the top level.
/// - Non-object inputs go into `"_input"`.
/// - Object configs are merged at the top level.
/// - Non-object configs are collected into `"_args"`.
pub(super) fn build_process_input(input: Option<&Value>, config: Option<&Value>) -> String {
    let mut obj = serde_json::Map::new();

    if let Some(val) = input {
        match val {
            Value::Object(map) => obj.extend(map.clone()),
            other => { obj.insert("_input".into(), other.clone()); }
        }
    }

    if let Some(cfg) = config {
        match cfg {
            Value::Object(map) => obj.extend(map.clone()),
            other => {
                let args = obj.entry("_args").or_insert(Value::Array(Vec::new()));
                if let Value::Array(arr) = args {
                    arr.push(other.clone());
                }
            }
        }
    }

    serde_json::to_string(&Value::Object(obj))
        .unwrap_or_else(|_| "{}".to_string())
}
