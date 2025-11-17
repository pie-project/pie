//! Service Architecture
//!
//! A framework for building asynchronous services that process commands via message-passing.
//! Each service runs in a daemon task and processes commands sequentially through a channel.
//!
//! # Architecture
//!
//! Each service consists of:
//! - A [`Service`] implementation that handles commands in a dedicated async task
//! - A command enum implementing [`ServiceCommand`] with all operations
//! - A static [`CommandDispatcher`] that routes commands to the service's task
//!
//! When [`Service::start()`] is called, it spawns a daemon task and initializes the
//! dispatcher. Commands can then be dispatched from anywhere using `.dispatch()`.
//!
//! # Usage Examples
//!
//! See `crate::kvs`, `crate::runtime`, and `crate::server` for complete implementations.

use std::sync::OnceLock;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task;

/// A service is a component that handles commands dispatched from other components.
pub trait Service
where
    Self: Sized + Send + 'static,
{
    /// The command type that this service handles.
    type Command: ServiceCommand;

    /// Handles a command.
    fn handle(&mut self, cmd: Self::Command) -> impl Future<Output = ()> + Send;

    /// Starts the service. Internally, it creates a channel and spawns a task that handles
    /// the commands. The `dispatcher` will be initialized with the sender of the channel that
    /// carries the dispatched commands.
    ///
    /// # Panics
    ///
    /// Panics if the service has already been started (i.e., the dispatcher is already initialized).
    fn start(mut self, dispatcher: &OnceLock<CommandDispatcher<Self::Command>>) {
        let (tx, mut rx) = unbounded_channel();

        task::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                self.handle(cmd).await;
            }
        });

        dispatcher
            .set(CommandDispatcher { tx })
            .map_err(|_| format!("Service {} already started", std::any::type_name::<Self>()))
            .unwrap();
    }
}

/// A command that can be dispatched to a service.
pub trait ServiceCommand: Send + 'static + Sized {
    /// The dispatcher that this command is sent through, which is essentially a sender
    /// of a channel.
    const DISPATCHER: &'static OnceLock<CommandDispatcher<Self>>;

    /// Sends this command to the service.
    fn dispatch(self) {
        // The dispatcher must be initialized before sending the command through
        // `Service::start()`. Sending through the internal channel must also
        // succeed. We unwrap to make sure these conditions are met.
        Self::DISPATCHER.get().unwrap().tx.send(self).unwrap();
    }
}

/// A dispatcher for commands, which is essentially a sender of a channel.
pub struct CommandDispatcher<T> {
    tx: UnboundedSender<T>,
}
