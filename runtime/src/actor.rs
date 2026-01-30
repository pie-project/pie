//! Actor Framework
//!
//! A lightweight actor model implementation for asynchronous message-passing services.
//! Each actor runs in a dedicated async task and processes messages sequentially.
//!
//! # Architecture
//!
//! - **Handle**: Trait for implementing message handlers
//! - **Actor**: Single actor address (for singletons)
//! - **Actors**: Table of actor addresses indexed by ID
//!
//! # Usage
//!
//! ## Singleton Actor
//! ```ignore
//! static ACTOR: LazyLock<Actor<MyMessage>> = LazyLock::new(Actor::new);
//! ACTOR.spawn::<MyHandler>();
//! ACTOR.send(msg)?;
//! ```
//!
//! ## Indexed Actors
//! ```ignore
//! static ACTOR: LazyLock<Actors<MyMessage>> = LazyLock::new(Actors::new);
//! let idx = ACTOR.spawn::<MyHandler>();
//! ACTOR.send(idx, msg)?;
//! ```

use std::future::Future;
use std::sync::OnceLock;
use thiserror::Error;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task;

// =============================================================================
// Common Types
// =============================================================================

/// Error returned when sending to an invalid actor.
#[derive(Debug, Error)]
pub enum SendError {
    #[error("Actor not spawned")]
    NotSpawned,
    #[error("Invalid actor index: {0}")]
    InvalidIndex(usize),
}

/// Trait for message handlers that process messages asynchronously.
pub trait Handle: Send + 'static {
    /// The message type this handler processes.
    type Message: Send + 'static;
    
    /// Creates a new instance of the handler.
    fn new() -> Self;
    
    /// Handles a message. Called sequentially for each message.
    fn handle(&mut self, msg: Self::Message) -> impl Future<Output = ()> + Send;
}

// =============================================================================
// Singleton Actor
// =============================================================================

/// A singleton actor address.
///
/// Use when you need exactly one actor instance (e.g., global services).
#[derive(Debug)]
pub struct Actor<Msg: Send + 'static> {
    tx: OnceLock<UnboundedSender<Msg>>,
}

impl<Msg: Send + 'static> Actor<Msg> {
    /// Creates a new empty actor.
    pub const fn new() -> Self {
        Self { tx: OnceLock::new() }
    }
    
    /// Spawns the actor.
    ///
    /// # Panics
    /// Panics if the actor has already been spawned.
    pub fn spawn<H>(&self)
    where
        H: Handle<Message = Msg>,
    {
        let mut handler = H::new();
        let (tx, mut rx) = unbounded_channel();

        self.tx
            .set(tx)
            .map_err(|_| "Actor already spawned")
            .unwrap();

        task::spawn(async move {
            while let Some(msg) = rx.recv().await {
                handler.handle(msg).await;
            }
        });
    }
    
    /// Spawns the actor using a factory function for custom initialization.
    ///
    /// # Panics
    /// Panics if the actor has already been spawned.
    pub fn spawn_with<H, F>(&self, factory: F)
    where
        H: Handle<Message = Msg>,
        F: FnOnce() -> H,
    {
        let mut handler = factory();
        let (tx, mut rx) = unbounded_channel();

        self.tx
            .set(tx)
            .map_err(|_| "Actor already spawned")
            .unwrap();

        task::spawn(async move {
            while let Some(msg) = rx.recv().await {
                handler.handle(msg).await;
            }
        });
    }
    
    /// Sends a message to the actor.
    pub fn send(&self, msg: Msg) -> Result<(), SendError> {
        self.tx
            .get()
            .ok_or(SendError::NotSpawned)?
            .send(msg)
            .map_err(|_| SendError::NotSpawned)
    }
    
    /// Returns true if the actor has been spawned.
    pub fn is_spawned(&self) -> bool {
        self.tx.get().is_some()
    }
}

// =============================================================================
// Indexed Actors
// =============================================================================

/// A table of actor addresses indexed by ID.
///
/// Use when you need one actor per model/context/etc.
#[derive(Debug)]
pub struct Actors<Msg: Send + 'static> {
    table: boxcar::Vec<UnboundedSender<Msg>>,
}

impl<Msg: Send + 'static> Actors<Msg> {
    /// Creates a new empty table.
    pub const fn new() -> Self {
        Self {
            table: boxcar::Vec::new(),
        }
    }
    
    /// Spawns a new actor and returns its index.
    pub fn spawn<H>(&self) -> usize
    where
        H: Handle<Message = Msg>,
    {
        let mut handler = H::new();
        let (tx, mut rx) = unbounded_channel();
        let idx = self.table.push(tx);

        task::spawn(async move {
            while let Some(msg) = rx.recv().await {
                handler.handle(msg).await;
            }
        });

        idx
    }
    
    /// Spawns an actor using a factory function for custom initialization.
    pub fn spawn_with<H, F>(&self, factory: F) -> usize
    where
        H: Handle<Message = Msg>,
        F: FnOnce() -> H,
    {
        let mut handler = factory();
        let (tx, mut rx) = unbounded_channel();
        let idx = self.table.push(tx);

        task::spawn(async move {
            while let Some(msg) = rx.recv().await {
                handler.handle(msg).await;
            }
        });

        idx
    }
    
    /// Sends a message to an actor by index.
    pub fn send(&self, idx: usize, msg: Msg) -> Result<(), SendError> {
        let tx = self
            .table
            .get(idx)
            .ok_or(SendError::InvalidIndex(idx))?;
        tx.send(msg).map_err(|_| SendError::InvalidIndex(idx))?;
        Ok(())
    }
    
    /// Returns the number of actors.
    pub fn len(&self) -> usize {
        self.table.count()
    }
    
    /// Returns true if no actors exist.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
