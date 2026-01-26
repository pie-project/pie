//! Context Service - Execution context management with KV cache state
//!
//! This module provides a model-specific actor for managing named execution
//! contexts with support for forking, joining, locking, and capacity management.

use dashmap::DashMap;
use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;
use anyhow::Result;

use crate::actor::{Handle, Actors, SendError};

/// Unique identifier for a context.
pub type ContextId = u64;
pub type LockId = u64;

/// Global table of context actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Spawns a new context actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn::<ContextActor>()
}

/// Messages for the context actor.
#[derive(Debug)]
pub enum Message {
    /// Creates a new context with the given name.
    Create {
        user_id: u32,
        name: String,
        response: oneshot::Sender<Result<ContextId>>,
    },

    /// Destroys a context.
    Destroy {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<Result<()>>,
    },

    /// Retrieves an existing context by name.
    Get {
        user_id: u32,
        name: String,
        response: oneshot::Sender<Option<ContextId>>,
    },

    /// Forks a context into a new one with the given name.
    Fork {
        id: ContextId,
        new_name: String,
        response: oneshot::Sender<Option<ContextId>>,
    },

    /// Merges another context into the target context.
    Join {
        target_id: ContextId,
        other_id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<Result<()>>,
    },

    /// Acquires a lock on the context.
    Lock {
        id: ContextId,
        response: oneshot::Sender<LockId>,
    },

    /// Releases the lock on the context.
    Unlock {
        id: ContextId,
        lock_id: LockId,
    },

    /// Grows the context capacity.
    Grow {
        id: ContextId,
        lock_id: LockId,
        size: u32,
        response: oneshot::Sender<Result<()>>,
    },

    /// Shrinks the context capacity.
    Shrink {
        id: ContextId,
        lock_id: LockId,
        size: u32,
        response: oneshot::Sender<Result<()>>,
    },
}

impl Message {
    /// Sends this message to the context actor for the given model.
    pub fn send(self, model_idx: usize) -> Result<(), SendError> {
        ACTOR.send(model_idx, self)
    }
}

/// Record of operations on a context (for lineage tracking).
#[derive(Debug, Clone)]
enum Record {
    Fill(Vec<u32>),
}

/// Internal representation of a context.
#[derive(Debug, Clone)]
struct Context {
    lineage: Vec<Record>,
    pages: Vec<usize>,
    last_page_len: usize,
    mutex: Option<LockId>,
}

impl Context {
    fn new() -> Self {
        Context {
            lineage: Vec::new(),
            pages: Vec::new(),
            last_page_len: 0,
            mutex: None,
        }
    }
}

/// The context actor manages named execution contexts.
#[derive(Debug)]
struct ContextActor {
    contexts: Arc<DashMap<ContextId, Context>>,
    name_to_id: Arc<DashMap<String, ContextId>>,
    next_id: Arc<AtomicU64>,
}

impl Handle for ContextActor {
    type Message = Message;

    fn new() -> Self {
        ContextActor {
            contexts: Arc::new(DashMap::new()),
            name_to_id: Arc::new(DashMap::new()),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Create { user_id: _, name, response } => {
                let id = self.next_id();
                let ctx = Context::new();
                self.contexts.insert(id, ctx);
                self.name_to_id.insert(name, id);
                let _ = response.send(Ok(id));
            }
            Message::Destroy { id, lock_id, response } => {
                let result = if let Some(ctx) = self.contexts.get(&id) {
                    if ctx.mutex == Some(lock_id) {
                        drop(ctx);
                        self.contexts.remove(&id);
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("Context not locked by this lock_id"))
                    }
                } else {
                    Err(anyhow::anyhow!("Context not found"))
                };
                let _ = response.send(result);
            }
            Message::Get { user_id: _, name, response } => {
                let id = self.name_to_id.get(&name).map(|v| *v.value());
                let _ = response.send(id);
            }
            Message::Fork { id, new_name, response } => {
                let result = if let Some(ctx) = self.contexts.get(&id) {
                    let new_id = self.next_id();
                    let mut new_ctx = ctx.value().clone();
                    new_ctx.mutex = None;
                    self.contexts.insert(new_id, new_ctx);
                    self.name_to_id.insert(new_name, new_id);
                    Some(new_id)
                } else {
                    None
                };
                let _ = response.send(result);
            }
            Message::Join { target_id, other_id, lock_id, response } => {
                let result = if let Some(target) = self.contexts.get(&target_id) {
                    if target.mutex != Some(lock_id) {
                        Err(anyhow::anyhow!("Target context not locked by this lock_id"))
                    } else if let Some(other) = self.contexts.get(&other_id) {
                        drop(other);
                        drop(target);
                        if let Some(mut target) = self.contexts.get_mut(&target_id) {
                            if let Some(other) = self.contexts.get(&other_id) {
                                target.pages.extend(other.pages.iter().cloned());
                            }
                        }
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("Other context not found"))
                    }
                } else {
                    Err(anyhow::anyhow!("Target context not found"))
                };
                let _ = response.send(result);
            }
            Message::Lock { id, response } => {
                let lock_id = self.next_id();
                if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if ctx.mutex.is_none() {
                        ctx.mutex = Some(lock_id);
                        let _ = response.send(lock_id);
                    } else {
                        let _ = response.send(0);
                    }
                } else {
                    let _ = response.send(0);
                }
            }
            Message::Unlock { id, lock_id } => {
                if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if ctx.mutex == Some(lock_id) {
                        ctx.mutex = None;
                    }
                }
            }
            Message::Grow { id, lock_id, size, response } => {
                let result = if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if ctx.mutex == Some(lock_id) {
                        let pages_needed = size as usize;
                        for _ in 0..pages_needed {
                            ctx.pages.push(0);
                        }
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("Context not locked by this lock_id"))
                    }
                } else {
                    Err(anyhow::anyhow!("Context not found"))
                };
                let _ = response.send(result);
            }
            Message::Shrink { id, lock_id, size, response } => {
                let result = if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if ctx.mutex == Some(lock_id) {
                        let pages_to_remove = (size as usize).min(ctx.pages.len());
                        let new_len = ctx.pages.len().saturating_sub(pages_to_remove);
                        ctx.pages.truncate(new_len);
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("Context not locked by this lock_id"))
                    }
                } else {
                    Err(anyhow::anyhow!("Context not found"))
                };
                let _ = response.send(result);
            }
        }
    }
}

impl ContextActor {
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}
