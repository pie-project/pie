//! Legacy Service Framework
//!
//! This module contains the legacy Service/ServiceCommand pattern for backwards compatibility.
//! New code should use the modern actor framework in `actor.rs` instead.

use std::future::Future;
use std::sync::OnceLock;

use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task;

/// Legacy: A singleton service that handles commands.
pub trait Service
where
    Self: Sized + Send + 'static,
{
    type Command: ServiceCommand;
    fn handle(&mut self, cmd: Self::Command) -> impl Future<Output = ()> + Send;

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

/// Legacy: A command that can be dispatched to a singleton service.
pub trait ServiceCommand: Send + 'static + Sized {
    const DISPATCHER: &'static OnceLock<CommandDispatcher<Self>>;

    fn dispatch(self) {
        Self::DISPATCHER.get().unwrap().tx.send(self).unwrap();
    }
}

/// Legacy: Dispatcher for singleton services.
pub struct CommandDispatcher<T> {
    pub(crate) tx: UnboundedSender<T>,
}
