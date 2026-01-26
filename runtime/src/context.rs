use dashmap::DashMap;
use std::sync::{Arc, LazyLock};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::task;
use anyhow::Result;

/// Unique identifier for a context.
pub type ContextId = u64;
pub type LockId = u64;

/// Model-indexed dispatcher for context services.
static CONTEXT_DISPATCHER: LazyLock<ContextDispatcher> = LazyLock::new(|| ContextDispatcher {
    services: boxcar::Vec::new(),
});

#[derive(Debug, Error)]
pub enum ContextDispatchError {
    #[error("Invalid model index: {0}")]
    InvalidModelIndex(usize),
}

#[derive(Debug)]
struct ContextDispatcher {
    services: boxcar::Vec<mpsc::UnboundedSender<Command>>,
}

/// Installs a new context service for the given model ID.
/// This is only called internally by model_new::install_model.
pub(super) fn install_service(model_id: usize) {
    let svc = ContextService::new();
    let (tx, mut rx) = mpsc::unbounded_channel();
    
    let idx = CONTEXT_DISPATCHER.services.push(tx);
    debug_assert_eq!(idx, model_id, "Context service ID mismatch");

    task::spawn(async move {
        let mut svc = svc;
        while let Some(cmd) = rx.recv().await {
            svc.handle(cmd).await;
        }
    });
}

/// Defines the set of operations available for the context service.
#[derive(Debug)]
pub enum Command {
    /// Creates a new context with the given name.
    /// Returns the ID of the newly created context.
    Create {
        user_id: u32,
        name: String,
        response: oneshot::Sender<Result<ContextId>>,
    },

    Destroy {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<Result<()>>,
    },

    /// Retrieves an existing context by name.
    /// Returns the context ID if found.
    Get {
        user_id: u32,
        name: String,
        response: oneshot::Sender<Option<ContextId>>,
    },
    /// Forks a context into a new one with the given name.
    /// Returns the ID of the newly forked context.
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

impl Command {
    /// Dispatches this command to the context service for the given model.
    pub fn dispatch(self, model_id: usize) -> Result<(), ContextDispatchError> {
        let tx = CONTEXT_DISPATCHER
            .services
            .get(model_id)
            .ok_or(ContextDispatchError::InvalidModelIndex(model_id))?;
        tx.send(self).unwrap();
        Ok(())
    }
}


#[derive(Debug, Clone)]
enum Record {
    Fill(Vec<u32>)
}

/// Internal representation of a context.
#[derive(Debug, Clone)]
struct Context {
    lineage: Vec<Record>,
    pages: Vec<usize>, // hash keys
    last_page_len: usize,
    mutex: Option<LockId>
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

/// The context service manages named execution contexts with support for
/// forking, joining, locking, and capacity management.
#[derive(Debug, Clone)]
struct ContextService {
    contexts: Arc<DashMap<ContextId, Context>>,
    name_to_id: Arc<DashMap<String, ContextId>>,
    next_id: Arc<std::sync::atomic::AtomicU64>,
}

impl ContextService {
    /// Creates a new, empty `ContextService`.
    fn new() -> Self {
        ContextService {
            contexts: Arc::new(DashMap::new()),
            name_to_id: Arc::new(DashMap::new()),
            next_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// Generates a new unique context ID.
    fn next_id(&self) -> ContextId {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    async fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::Create { user_id: _, name, response } => {
                let id = self.next_id();
                let ctx = Context::new();
                self.contexts.insert(id, ctx);
                self.name_to_id.insert(name, id);
                let _ = response.send(Ok(id));
            }
            Command::Destroy { id, lock_id, response } => {
                let result = if let Some(ctx) = self.contexts.get(&id) {
                    if ctx.mutex == Some(lock_id) {
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
            Command::Get { user_id: _, name, response } => {
                let id = self.name_to_id.get(&name).map(|v| *v.value());
                let _ = response.send(id);
            }
            Command::Fork {
                id,
                new_name,
                response,
            } => {
                let result = if let Some(ctx) = self.contexts.get(&id) {
                    let new_id = self.next_id();
                    let mut new_ctx = ctx.value().clone();
                    new_ctx.mutex = None; // New fork is unlocked
                    self.contexts.insert(new_id, new_ctx);
                    self.name_to_id.insert(new_name, new_id);
                    Some(new_id)
                } else {
                    None
                };
                let _ = response.send(result);
            }
            Command::Join {
                target_id,
                other_id,
                lock_id,
                response,
            } => {
                let result = if let Some(target) = self.contexts.get(&target_id) {
                    if target.mutex != Some(lock_id) {
                        Err(anyhow::anyhow!("Target context not locked by this lock_id"))
                    } else if let Some(other) = self.contexts.get(&other_id) {
                        // Merge pages from other into target
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
            Command::Lock { id, response } => {
                let lock_id = self.next_id(); // Use next_id as lock_id generator
                if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if ctx.mutex.is_none() {
                        ctx.mutex = Some(lock_id);
                        let _ = response.send(lock_id);
                    } else {
                        // Already locked, send 0 to indicate failure
                        let _ = response.send(0);
                    }
                } else {
                    let _ = response.send(0);
                }
            }
            Command::Unlock { id, lock_id } => {
                if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if ctx.mutex == Some(lock_id) {
                        ctx.mutex = None;
                    }
                }
            }
            Command::Grow { id, lock_id, size, response } => {
                let result = if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if ctx.mutex == Some(lock_id) {
                        // Add new pages as needed
                        let pages_needed = size as usize;
                        for _ in 0..pages_needed {
                            ctx.pages.push(0); // Placeholder page
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
            Command::Shrink { id, lock_id, size, response } => {
                let result = if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if ctx.mutex == Some(lock_id) {
                        let pages_to_remove = (size as usize).min(ctx.pages.len());
                        ctx.pages.truncate(ctx.pages.len().saturating_sub(pages_to_remove));
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
