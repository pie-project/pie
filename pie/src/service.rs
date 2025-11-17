use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{OnceLock, RwLock};
use thiserror::Error;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task;

// Common driver routines

static LEGACY_SERVICE_DISPATCHER: OnceLock<LegacyServiceDispatcher> = OnceLock::new();

pub fn install_legacy_service<T>(name: &str, driver: T) -> Option<usize>
where
    T: LegacyService + 'static + Send,
{
    // check if the dispatcher is initialized
    let dispatcher = LEGACY_SERVICE_DISPATCHER.get_or_init(|| LegacyServiceDispatcher {
        maps: RwLock::new(HashMap::new()),
        channels: boxcar::Vec::new(),
    });

    dispatcher.install(name, driver)
}

pub fn dispatch_legacy<C>(service_id: usize, cmd: C) -> Result<(), LegacyServiceError>
where
    C: Any + Send + Sync + 'static,
{
    LEGACY_SERVICE_DISPATCHER
        .get()
        .expect("Service is not initialized")
        .dispatch(service_id, cmd)
}

pub fn get_legacy_service_id(name: &str) -> Option<usize> {
    LEGACY_SERVICE_DISPATCHER
        .get()
        .expect("Service not found")
        .get_service_id(name)
}

#[derive(Debug, Error)]
pub enum LegacyServiceError {
    #[error("Driver '{0}' not found")]
    DriverNotFound(String),

    #[error("Invalid driver index: {0}")]
    InvalidDriverIndex(usize),
}
//#[async_trait]
pub trait LegacyService: Send {
    type Command: Send + Sync + 'static;

    fn handle(&mut self, cmd: Self::Command) -> impl Future<Output = ()> + Send;
}

pub type AnyCommand = Box<dyn Any + Send + Sync>;

#[derive(Debug)]
pub struct LegacyServiceDispatcher {
    maps: RwLock<HashMap<String, usize>>,

    // Note: Using `boxcar::Vec` for performance
    // boxcar::Vec<> took:  14.79875ms
    // RwLock<Vec<>> took: 54.038458ms
    channels: boxcar::Vec<UnboundedSender<AnyCommand>>,
}

impl LegacyServiceDispatcher {
    pub fn get_service_id(&self, name: &str) -> Option<usize> {
        self.maps.read().unwrap().get(name).copied()
    }

    pub fn install<T>(&self, name: &str, mut driver: T) -> Option<usize>
    where
        T: LegacyService + 'static + Send,
    {
        let (tx, mut rx) = unbounded_channel();

        // first, make sure the name is not already registered
        if self.get_service_id(name).is_some() {
            return None;
        }

        self.channels.push(tx);
        let service_id = self.channels.count() - 1;

        self.maps
            .write()
            .unwrap()
            .insert(name.to_string(), service_id);

        task::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                driver.handle(*cmd.downcast().unwrap()).await;
            }
        });

        Some(service_id)
    }

    pub fn dispatch<C>(&self, service_id: usize, cmd: C) -> Result<(), LegacyServiceError>
    where
        C: Any + Send + Sync + 'static,
    {
        let cmd = Box::new(cmd);

        self.channels
            .get(service_id)
            .ok_or(LegacyServiceError::InvalidDriverIndex(service_id))?
            .send(cmd)
            .unwrap();

        Ok(())
    }

    pub fn dispatch_with<C>(&self, name: &str, cmd: C) -> Result<(), LegacyServiceError>
    where
        C: Any + Send + Sync + 'static,
    {
        let service_id = self
            .get_service_id(name)
            .ok_or(LegacyServiceError::DriverNotFound(name.to_string()))?;

        self.dispatch(service_id, cmd)?;

        Ok(())
    }
}

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
