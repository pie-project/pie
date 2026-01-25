use dashmap::DashMap;
use std::sync::{Arc, LazyLock};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::task;

/// Unique identifier for a context.
pub type ContextId = u64;

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
        name: String,
        response: oneshot::Sender<ContextId>,
    },
    /// Retrieves an existing context by name.
    /// Returns the context ID if found.
    Get {
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
        response: oneshot::Sender<bool>,
    },
    /// Acquires a lock on the context.
    Lock {
        id: ContextId,
        response: oneshot::Sender<bool>,
    },
    /// Releases the lock on the context.
    Unlock {
        id: ContextId,
        response: oneshot::Sender<bool>,
    },
    /// Grows the context capacity.
    Grow {
        id: ContextId,
        size: u32,
        response: oneshot::Sender<bool>,
    },
    /// Shrinks the context capacity.
    Shrink {
        id: ContextId,
        size: u32,
        response: oneshot::Sender<bool>,
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

/// Internal representation of a context.
#[derive(Debug, Clone)]
struct Context {
    name: String,
    capacity: u32,
    locked: bool,
}

impl Context {
    fn new(name: String) -> Self {
        Context {
            name,
            capacity: 0,
            locked: false,
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
            Command::Create { name, response } => {
                let id = self.next_id();
                let ctx = Context::new(name.clone());
                self.contexts.insert(id, ctx);
                self.name_to_id.insert(name, id);
                let _ = response.send(id);
            }
            Command::Get { name, response } => {
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
                    new_ctx.name = new_name.clone();
                    new_ctx.locked = false;
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
                response,
            } => {
                let success = if self.contexts.contains_key(&target_id)
                    && self.contexts.contains_key(&other_id)
                {
                    // Merge capacity from other into target
                    if let Some(other) = self.contexts.get(&other_id) {
                        if let Some(mut target) = self.contexts.get_mut(&target_id) {
                            target.capacity += other.capacity;
                        }
                    }
                    true
                } else {
                    false
                };
                let _ = response.send(success);
            }
            Command::Lock { id, response } => {
                let success = if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    if !ctx.locked {
                        ctx.locked = true;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                let _ = response.send(success);
            }
            Command::Unlock { id, response } => {
                let success = if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    ctx.locked = false;
                    true
                } else {
                    false
                };
                let _ = response.send(success);
            }
            Command::Grow { id, size, response } => {
                let success = if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    ctx.capacity = ctx.capacity.saturating_add(size);
                    true
                } else {
                    false
                };
                let _ = response.send(success);
            }
            Command::Shrink { id, size, response } => {
                let success = if let Some(mut ctx) = self.contexts.get_mut(&id) {
                    ctx.capacity = ctx.capacity.saturating_sub(size);
                    true
                } else {
                    false
                };
                let _ = response.send(success);
            }
        }
    }
}
