use super::service::{CommandDispatcher, Service, ServiceCommand};
use dashmap::DashMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::oneshot;

/// The sender of the command channel, which is used to send commands to the
/// handler task.
static COMMAND_DISPATCHER: OnceLock<CommandDispatcher<Command>> = OnceLock::new();

/// Starts the key-value store service. A daemon task will be spawned to handle the
/// commands dispatched from other services.
pub fn start_service() {
    let kvs = KeyValueStore::new();
    kvs.start(&COMMAND_DISPATCHER);
}

/// Defines the set of operations available for the key-value store.
#[derive(Debug)]
pub enum Command {
    /// Retrieves a value associated with a key.
    /// The result is sent back as an `Option<String>`.
    Get {
        key: String,
        response: oneshot::Sender<Option<String>>,
    },
    /// Inserts or updates a key-value pair.
    /// The `oneshot::Sender` is used to signal completion.
    Set { key: String, value: String },
    /// Removes a key-value pair.
    /// The `oneshot::Sender` is used to signal completion.
    Delete { key: String },
    /// Checks if a key exists in the store.
    /// The result is sent back as a `bool`.
    Exists {
        key: String,
        response: oneshot::Sender<bool>,
    },
    /// Retrieves a list of all keys currently in the store.
    /// The result is sent back as a `Vec<String>`.
    ListKeys {
        response: oneshot::Sender<Vec<String>>,
    },
}

impl ServiceCommand for Command {
    const DISPATCHER: &'static OnceLock<CommandDispatcher<Self>> = &COMMAND_DISPATCHER;
}

/// An in-memory key-value store service.
///
/// It uses a `DashMap` for concurrent, lock-free reads and writes,
/// making it suitable for a multi-threaded, asynchronous environment.
#[derive(Debug, Clone)]
struct KeyValueStore {
    store: Arc<DashMap<String, String>>,
}

impl KeyValueStore {
    /// Creates a new, empty `KeyValueStore`.
    fn new() -> Self {
        KeyValueStore {
            store: Arc::new(DashMap::new()),
        }
    }
}

impl Service for KeyValueStore {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::Get { key, response } => {
                let value = self.store.get(&key).map(|v| v.value().clone());
                let _ = response.send(value);
            }
            Command::Set { key, value } => {
                self.store.insert(key, value);
            }
            Command::Delete { key } => {
                self.store.remove(&key);
            }
            Command::Exists { key, response } => {
                let exists = self.store.contains_key(&key);
                let _ = response.send(exists);
            }
            Command::ListKeys { response } => {
                let keys: Vec<String> =
                    self.store.iter().map(|entry| entry.key().clone()).collect();
                let _ = response.send(keys);
            }
        }
    }
}
