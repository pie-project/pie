use anyhow::Result;
use async_trait::async_trait;

/// The base trait for all actors in the system.
#[async_trait]
pub trait Actor: Send + Sync {
    /// Creates a new actor instance for the given user.
    async fn create(&self, username: String, uid: String) -> Result<()>;

    /// Destroys the actor instance for the given user.
    async fn destroy(&self, username: String, uid: String) -> Result<()>;
}

/// Actor service for managing adapters.
pub struct AdapterActor;

#[async_trait]
impl Actor for AdapterActor {
    async fn create(&self, _username: String, _uid: String) -> Result<()> {
        Ok(())
    }

    async fn destroy(&self, _username: String, _uid: String) -> Result<()> {
        Ok(())
    }
}

/// Actor service for managing context.
pub struct ContextActor;

#[async_trait]
impl Actor for ContextActor {
    async fn create(&self, _username: String, _uid: String) -> Result<()> {
        Ok(())
    }

    async fn destroy(&self, _username: String, _uid: String) -> Result<()> {
        Ok(())
    }
}

/// Actor service for managing optimizations.
pub struct OptimizerActor;

#[async_trait]
impl Actor for OptimizerActor {
    async fn create(&self, _username: String, _uid: String) -> Result<()> {
        Ok(())
    }

    async fn destroy(&self, _username: String, _uid: String) -> Result<()> {
        Ok(())
    }
}
