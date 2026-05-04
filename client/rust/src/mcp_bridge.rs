//! Local bridge from this client to MCP servers.
//!
//! When the engine asks us to relay an MCP call (because some inferlet called
//! `mcp::client::session::call_tool` etc.), we look up the named server here
//! and forward the call as JSON-RPC over the server's transport.
//!
//! Currently only the `stdio` transport is implemented: we spawn a child
//! process, write line-delimited JSON-RPC requests to its stdin, and read
//! line-delimited JSON-RPC responses from its stdout. One child per
//! registered name; many inferlets multiplex over the single connection
//! using distinct request ids.

use anyhow::{Context, Result, anyhow, bail};
use dashmap::DashMap;
use serde_json::{Value, json};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC error decoded from an MCP server response.
#[derive(Debug, Clone)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for JsonRpcError {}

/// Per-`Client` registry of locally-spawned MCP servers.
pub struct BridgeRegistry {
    servers: DashMap<String, Arc<StdioServer>>,
}

impl std::fmt::Debug for BridgeRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<String> = self.servers.iter().map(|r| r.key().clone()).collect();
        f.debug_struct("BridgeRegistry").field("servers", &names).finish()
    }
}

impl BridgeRegistry {
    pub fn new() -> Self {
        Self { servers: DashMap::new() }
    }

    /// Spawn a stdio MCP server, complete the `initialize` handshake, and
    /// publish it under `name`. Fails if `name` is already registered or if
    /// the handshake doesn't succeed.
    pub async fn register_stdio(
        &self,
        name: &str,
        command: &str,
        args: &[String],
    ) -> Result<()> {
        if self.servers.contains_key(name) {
            bail!("MCP server '{}' already registered", name);
        }
        let server = StdioServer::spawn(name, command, args).await?;
        server.handshake().await
            .with_context(|| format!("MCP handshake with '{}' failed", name))?;
        self.servers.insert(name.to_string(), Arc::new(server));
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<StdioServer>> {
        self.servers.get(name).map(|r| Arc::clone(r.value()))
    }
}

impl Default for BridgeRegistry {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// StdioServer
// =============================================================================

/// State shared between an MCP server's I/O tasks and the public handle.
struct StdioInner {
    name: String,
    stdin_tx: tokio::sync::mpsc::UnboundedSender<String>,
    pending: DashMap<i64, oneshot::Sender<std::result::Result<Value, JsonRpcError>>>,
    next_id: AtomicI64,
    dead: AtomicBool,
}

/// Aborts a `JoinHandle` when dropped. Used to tie task lifetimes to the
/// outer handle so that dropping the registry entry tears down the child
/// without leaking tasks.
struct AbortOnDrop(JoinHandle<()>);
impl Drop for AbortOnDrop { fn drop(&mut self) { self.0.abort(); } }

/// A connection to a single stdio MCP server. Cheap to clone (`Arc`), all
/// methods route through the multiplexed JSON-RPC channel.
pub struct StdioServer {
    inner: Arc<StdioInner>,
    // Field order matters: Rust drops fields top-to-bottom, so on Drop we
    // first abort the supervisor (whose `Child` is `kill_on_drop` → process
    // dies), which lets reader/writer end naturally on EOF.
    _supervisor: AbortOnDrop,
    _reader: AbortOnDrop,
    _writer: AbortOnDrop,
    _stderr: AbortOnDrop,
}

impl StdioServer {
    async fn spawn(name: &str, command: &str, args: &[String]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server '{}': {}", name, command))?;

        let stdin = child.stdin.take().context("MCP child has no stdin")?;
        let stdout = child.stdout.take().context("MCP child has no stdout")?;
        let stderr = child.stderr.take().context("MCP child has no stderr")?;

        let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let inner = Arc::new(StdioInner {
            name: name.to_string(),
            stdin_tx: line_tx,
            pending: DashMap::new(),
            next_id: AtomicI64::new(1),
            dead: AtomicBool::new(false),
        });

        // stdin writer
        let writer = tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(line) = line_rx.recv().await {
                if stdin.write_all(line.as_bytes()).await.is_err() { break; }
                if stdin.write_all(b"\n").await.is_err() { break; }
                if stdin.flush().await.is_err() { break; }
            }
        });

        // stdout reader — routes JSON-RPC responses to pending callers
        let reader_inner = Arc::clone(&inner);
        let reader = tokio::spawn(async move {
            let mut br = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match br.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                let v: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[mcp:{}] non-JSON line ({}): {}", reader_inner.name, e, trimmed);
                        continue;
                    }
                };
                // We only handle responses (id present + result|error).
                // Server-initiated requests/notifications are ignored for now.
                let id = match v.get("id").and_then(Value::as_i64) {
                    Some(i) => i,
                    None => continue,
                };
                if let Some((_, sender)) = reader_inner.pending.remove(&id) {
                    let outcome = if let Some(err) = v.get("error") {
                        let code = err.get("code").and_then(Value::as_i64).unwrap_or(-32000);
                        let message = err.get("message").and_then(Value::as_str)
                            .unwrap_or("MCP error").to_string();
                        let data = err.get("data").cloned();
                        Err(JsonRpcError { code, message, data })
                    } else {
                        Ok(v.get("result").cloned().unwrap_or(Value::Null))
                    };
                    let _ = sender.send(outcome);
                } else {
                    eprintln!("[mcp:{}] response for unknown id {}", reader_inner.name, id);
                }
            }
            // EOF / error: mark dead and fail any outstanding pendings.
            reader_inner.dead.store(true, Ordering::SeqCst);
            reader_inner.pending.clear();
        });

        // stderr drain — surface server errors to the client's stderr
        let name_for_err = name.to_string();
        let stderr_task = tokio::spawn(async move {
            let mut br = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match br.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                eprint!("[mcp:{}] {}", name_for_err, line);
            }
        });

        // Supervisor — owns Child, reaps on exit. kill_on_drop ensures the
        // process dies if the supervisor task is aborted.
        let supervisor = tokio::spawn(async move {
            let _ = child.wait().await;
        });

        Ok(Self {
            inner,
            _supervisor: AbortOnDrop(supervisor),
            _reader: AbortOnDrop(reader),
            _writer: AbortOnDrop(writer),
            _stderr: AbortOnDrop(stderr_task),
        })
    }

    /// Perform the MCP `initialize` handshake and send `initialized`.
    async fn handshake(&self) -> Result<()> {
        let init_params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "pie-client",
                "version": env!("CARGO_PKG_VERSION"),
            },
        });
        self.call("initialize", init_params).await
            .map_err(|e| anyhow!("{}", e))?;
        // Notification — no response expected.
        self.notify("notifications/initialized", json!({}))?;
        Ok(())
    }

    /// Send a JSON-RPC request and await the matching response.
    pub async fn call(
        &self,
        method: &str,
        params: Value,
    ) -> std::result::Result<Value, JsonRpcError> {
        if self.inner.dead.load(Ordering::SeqCst) {
            return Err(JsonRpcError {
                code: -32000,
                message: format!("MCP server '{}' is no longer running", self.inner.name),
                data: None,
            });
        }
        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.inner.pending.insert(id, tx);
        let req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        if self.inner.stdin_tx.send(req.to_string()).is_err() {
            self.inner.pending.remove(&id);
            return Err(JsonRpcError {
                code: -32000,
                message: format!("MCP server '{}' input closed", self.inner.name),
                data: None,
            });
        }
        match rx.await {
            Ok(outcome) => outcome,
            Err(_) => {
                self.inner.pending.remove(&id);
                Err(JsonRpcError {
                    code: -32000,
                    message: format!("MCP server '{}' died before response", self.inner.name),
                    data: None,
                })
            }
        }
    }

    /// Send a JSON-RPC notification (no id, no response).
    fn notify(&self, method: &str, params: Value) -> Result<()> {
        let req = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.inner.stdin_tx.send(req.to_string())
            .map_err(|_| anyhow!("MCP server '{}' input closed", self.inner.name))?;
        Ok(())
    }
}
