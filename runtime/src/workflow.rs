//! Workflow Composition Layer
//!
//! Defines a JSON DSL for composing processes into DAGs, and a tree-walking
//! async interpreter that evaluates the resulting expression tree.
//!
//! # Architecture
//!
//! ```text
//! JSON ─► Expr (recursive DSL) ─► Executor (async tree-walk)
//!                                      │
//!                                      ├─ Process → spawn & await
//!                                      ├─ Pipe    → sequential chain
//!                                      ├─ Fork    → concurrent (FuturesUnordered)
//!                                      ├─ Map     → eval over, fan-out
//!                                      ├─ Fold    → eval over, serial chain
//!                                      ├─ Cond    → eval predicate, chosen branch
//!                                      └─ Iterate → loop body→until
//! ```
//!
//! The `Workflow` actor manages lifecycle (cancel, status queries).
//! The `Executor` does the actual evaluation as a spawned async task.

mod expr;
mod executor;

use std::sync::{Arc, LazyLock};

use anyhow::{anyhow, Result, bail};
use serde::Serialize;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::process::{ProcessId, ProcessEvent};
use crate::server::{self, ClientId};
use crate::service::{ServiceMap, ServiceHandler};

pub(crate) use expr::Expr;
use executor::{Executor, value_to_string};

pub(crate) type WorkflowId = Uuid;

// =============================================================================
// Globals
// =============================================================================

static SERVICES: LazyLock<ServiceMap<WorkflowId, Message>> =
    LazyLock::new(ServiceMap::new);

// =============================================================================
// Public API
// =============================================================================

/// Submit a workflow defined by a JSON expression string.
/// Returns a `WorkflowId` and a channel to receive the final result.
pub async fn submit(
    username: &str,
    json: &str,
    client_id: Option<ClientId>,
) -> Result<(WorkflowId, oneshot::Receiver<Result<String, String>>)> {
    let expr: Expr = serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("Invalid workflow JSON: {e}"))?;

    // Validate non-empty
    if matches!(expr, Expr::Pipe { ref stages } if stages.is_empty()) {
        bail!("Workflow has no executable steps");
    }

    let id = WorkflowId::new_v4();
    let (result_tx, result_rx) = oneshot::channel();
    let executor = Arc::new(Executor::new(id, username.to_string()));

    SERVICES.spawn(id, {
        let executor = executor.clone();
        let username = username.to_string();
        move || Workflow {
            id,
            username,
            expr: Some(expr),
            executor,
            result_tx: Some(result_tx),
            client_id,
            events: Vec::new(),
        }
    })?;

    Ok((id, result_rx))
}

/// Cancel a running workflow.
pub fn cancel(id: &WorkflowId) -> Result<()> {
    SERVICES.send(id, Message::Cancel)
}

/// Query the stats of a workflow.
pub async fn get_stats(id: &WorkflowId) -> Result<WorkflowStats> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(id, Message::GetStats { response: tx })?;
    Ok(rx.await?)
}

/// Forward a process event to the workflow that spawned it.
pub(crate) fn forward_event(id: WorkflowId, pid: ProcessId, event: ProcessEvent) -> Result<()> {
    SERVICES.send(&id, Message::ProcessEvent { pid, event })
}

/// Attach a client to a workflow.
pub async fn attach(id: &WorkflowId, client_id: ClientId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(id, Message::AttachClient { client_id, response: tx })?;
    rx.await?
}

/// Detach the current client from a workflow (fire-and-forget).
pub fn detach(id: &WorkflowId) {
    let _ = SERVICES.send(id, Message::DetachClient);
}

/// Get the username that submitted a workflow.
pub async fn get_username(id: &WorkflowId) -> Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(id, Message::GetUsername { response: tx })?;
    Ok(rx.await?)
}

// =============================================================================
// Messages & Status
// =============================================================================

enum Message {
    /// An event forwarded from a child process.
    ProcessEvent { pid: ProcessId, event: ProcessEvent },
    /// Cancel the entire workflow.
    Cancel,
    /// Query workflow status.
    GetStats { response: oneshot::Sender<WorkflowStats> },
    /// Attach a client to receive events.
    AttachClient { client_id: ClientId, response: oneshot::Sender<Result<()>> },
    /// Detach the current client.
    DetachClient,
    /// Query the submitter username.
    GetUsername { response: oneshot::Sender<String> },
}

/// External stats snapshot for a workflow.
#[derive(Debug, Clone, Serialize)]
pub enum WorkflowStats {
    Running {
        total_steps: usize,
        completed_steps: usize,
    },
    Completed(String),
    Failed(String),
    Cancelled,
}

// =============================================================================
// Actor
// =============================================================================

/// The workflow actor. Manages lifecycle around an `Executor`.
struct Workflow {
    id: WorkflowId,
    username: String,
    /// Expression tree — taken in `started()` and moved to the eval task.
    expr: Option<Expr>,
    executor: Arc<Executor>,
    result_tx: Option<oneshot::Sender<Result<String, String>>>,
    /// Currently attached client, if any.
    client_id: Option<ClientId>,
    /// Accumulated process events (stdout, stderr, messages, etc.).
    events: Vec<(ProcessId, ProcessEvent)>,
}

impl ServiceHandler for Workflow {
    type Message = Message;

    async fn started(&mut self) {
        tracing::info!("Workflow {} started", self.id);

        let expr = self.expr.take().expect("expr already taken");
        let executor = self.executor.clone();
        let wf_id = self.id;
        let result_tx = self.result_tx.take();

        tokio::spawn(async move {
            let result = executor.run(&expr).await;
            let string_result = match result {
                Ok(ref val) => Ok(value_to_string(val)),
                Err(e) => Err(e),
            };
            if let Some(tx) = result_tx {
                let _ = tx.send(string_result);
            }
            SERVICES.remove(&wf_id);
        });
    }

    async fn stopped(&mut self) {
        tracing::info!("Workflow {} stopped", self.id);
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::ProcessEvent { pid, event } => {
                // Deliver to attached client
                if let Some(client_id) = self.client_id {
                    if server::send_event(client_id, pid, &event).is_err() {
                        self.client_id = None;
                    }
                } else {
                    self.events.push((pid, event));
                }
            }
            Message::Cancel => {
                self.executor.cancel.cancel();
            }
            Message::GetStats { response } => {
                let status = if self.executor.cancel.is_cancelled() {
                    WorkflowStats::Cancelled
                } else {
                    WorkflowStats::Running {
                        total_steps: self.executor.total(),
                        completed_steps: self.executor.completed(),
                    }
                };
                let _ = response.send(status);
            }
            Message::AttachClient { client_id, response } => {
                if self.client_id.is_some() {
                    let _ = response.send(Err(anyhow!("already attached")));
                } else {
                    self.client_id = Some(client_id);
                    // Flush buffered events to the newly attached client
                    for (pid, event) in &self.events {
                        if server::send_event(client_id, *pid, event).is_err() {
                            self.client_id = None;
                            break;
                        }
                    }
                    self.events.clear();
                    let _ = response.send(Ok(()));
                }
            }
            Message::DetachClient => {
                self.client_id = None;
            }
            Message::GetUsername { response } => {
                let _ = response.send(self.username.clone());
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::executor::build_process_input;
    use serde_json::Value;

    #[test]
    fn parse_process_expr() {
        let json = r#"{"type": "process", "program_name": "echo"}"#;
        let expr: Expr = serde_json::from_str(json).unwrap();
        match expr {
            Expr::Process { program_name, .. } => {
                assert_eq!(program_name, "echo");
            }
            _ => panic!("expected Process"),
        }
    }

    #[test]
    fn parse_pipe_expr() {
        let json = r#"{
            "type": "pipe",
            "stages": [
                {"type": "process", "program_name": "tokenize"},
                {"type": "process", "program_name": "generate"}
            ]
        }"#;
        let expr: Expr = serde_json::from_str(json).unwrap();
        match expr {
            Expr::Pipe { stages } => assert_eq!(stages.len(), 2),
            _ => panic!("expected Pipe"),
        }
    }

    #[test]
    fn parse_map_expr() {
        let json = r#"{
            "type": "map",
            "function": {"type": "process", "program_name": "summarize"},
            "over": {"type": "literal", "value": ["doc1", "doc2", "doc3"]}
        }"#;
        let expr: Expr = serde_json::from_str(json).unwrap();
        match expr {
            Expr::Map { function, over: _ } => {
                assert!(matches!(*function, Expr::Process { .. }));
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn parse_cond_expr() {
        let json = r#"{
            "type": "cond",
            "predicate": {"type": "process", "program_name": "check"},
            "then": {"type": "process", "program_name": "yes"},
            "otherwise": {"type": "process", "program_name": "no"}
        }"#;
        let expr: Expr = serde_json::from_str(json).unwrap();
        assert!(matches!(expr, Expr::Cond { .. }));
    }

    #[test]
    fn parse_iterate_expr() {
        let json = r#"{
            "type": "iterate",
            "body": {"type": "process", "program_name": "step"},
            "until": {"type": "process", "program_name": "check"}
        }"#;
        let expr: Expr = serde_json::from_str(json).unwrap();
        assert!(matches!(expr, Expr::Iterate { .. }));
    }

    #[test]
    fn parse_fold_expr() {
        let json = r#"{
            "type": "fold",
            "function": {"type": "process", "program_name": "reduce"},
            "init": {"type": "literal", "value": 0},
            "over": {"type": "literal", "value": ["x", "y", "z"]}
        }"#;
        let expr: Expr = serde_json::from_str(json).unwrap();
        assert!(matches!(expr, Expr::Fold { .. }));
    }

    #[test]
    fn parse_take_expr() {
        let json = r#"{
            "type": "pipe",
            "stages": [
                {"type": "fork", "branches": [
                    {"type": "process", "program_name": "a"},
                    {"type": "process", "program_name": "b"}
                ]},
                {"type": "take", "k": 1}
            ]
        }"#;
        let expr: Expr = serde_json::from_str(json).unwrap();
        match expr {
            Expr::Pipe { stages } => {
                assert_eq!(stages.len(), 2);
                assert!(matches!(stages[0], Expr::Fork { .. }));
                assert!(matches!(stages[1], Expr::Take { k: 1 }));
            }
            _ => panic!("expected Pipe"),
        }
    }

    #[test]
    fn build_process_input_object_source() {
        let input = serde_json::json!({"key": "val"});
        let config = serde_json::json!("extra");
        let result = build_process_input(Some(&input), Some(&config));
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["key"], "val");
        assert_eq!(parsed["_args"][0], "extra");
    }

    #[test]
    fn build_process_input_non_object_source() {
        let input = serde_json::json!([1, 2, 3]);
        let result = build_process_input(Some(&input), None);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["_input"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn build_process_input_empty() {
        let result = build_process_input(None, None);
        assert_eq!(result, "{}");
    }

    #[test]
    fn build_process_input_object_config() {
        let config = serde_json::json!({"model": "gpt-4"});
        let result = build_process_input(None, Some(&config));
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["model"], "gpt-4");
    }
}
