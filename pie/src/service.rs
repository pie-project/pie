use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{OnceLock, RwLock};
use thiserror::Error;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task;

// Common driver routines

static SERVICE_DISPATCHER: OnceLock<ServiceDispatcher> = OnceLock::new();

pub fn install_service<T>(name: &str, driver: T) -> Option<usize>
where
    T: Service + 'static + Send,
{
    // check if the dispatcher is initialized
    let dispatcher = SERVICE_DISPATCHER.get_or_init(|| ServiceDispatcher {
        maps: RwLock::new(HashMap::new()),
        channels: boxcar::Vec::new(),
    });

    dispatcher.install(name, driver)
}

pub fn dispatch<C>(service_id: usize, cmd: C) -> Result<(), ServiceError>
where
    C: Any + Send + Sync + 'static,
{
    SERVICE_DISPATCHER
        .get()
        .expect("Service is not initialized")
        .dispatch(service_id, cmd)
}

pub fn get_service_id(name: &str) -> Option<usize> {
    SERVICE_DISPATCHER
        .get()
        .expect("Service not found")
        .get_service_id(name)
}

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("Driver '{0}' not found")]
    DriverNotFound(String),

    #[error("Invalid driver index: {0}")]
    InvalidDriverIndex(usize),
}
//#[async_trait]
pub trait Service: Send {
    type Command: Send + Sync + 'static;

    fn handle(&mut self, cmd: Self::Command) -> impl Future<Output = ()> + Send;
}

pub type AnyCommand = Box<dyn Any + Send + Sync>;

#[derive(Debug)]
pub struct ServiceDispatcher {
    maps: RwLock<HashMap<String, usize>>,

    // Note: Using `boxcar::Vec` for performance
    // boxcar::Vec<> took:  14.79875ms
    // RwLock<Vec<>> took: 54.038458ms
    channels: boxcar::Vec<UnboundedSender<AnyCommand>>,
}

impl ServiceDispatcher {
    pub fn get_service_id(&self, name: &str) -> Option<usize> {
        self.maps.read().unwrap().get(name).copied()
    }

    pub fn install<T>(&self, name: &str, mut driver: T) -> Option<usize>
    where
        T: Service + 'static + Send,
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

    pub fn dispatch<C>(&self, service_id: usize, cmd: C) -> Result<(), ServiceError>
    where
        C: Any + Send + Sync + 'static,
    {
        let cmd = Box::new(cmd);

        self.channels
            .get(service_id)
            .ok_or(ServiceError::InvalidDriverIndex(service_id))?
            .send(cmd)
            .unwrap();

        Ok(())
    }

    pub fn dispatch_with<C>(&self, name: &str, cmd: C) -> Result<(), ServiceError>
    where
        C: Any + Send + Sync + 'static,
    {
        let service_id = self
            .get_service_id(name)
            .ok_or(ServiceError::DriverNotFound(name.to_string()))?;

        self.dispatch(service_id, cmd)?;

        Ok(())
    }
}
