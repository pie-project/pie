//! # Backend Module
//!
//! Placeholder service for backend operations.
//!
//! ## Architecture
//!
//! The Backend follows the Service-Actor pattern:
//! - **Backend** (singleton) - Manages backend operations
//!

use std::sync::LazyLock;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::service::{Service, ServiceHandler};

// =============================================================================
// Public API
// =============================================================================

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the backend service.
pub fn spawn() {
    SERVICE.spawn::<Backend, _>(Backend::new).expect("Backend already spawned");
}

/// Placeholder: Checks if the backend is healthy.
pub async fn health_check() -> Result<bool> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::HealthCheck { response: tx })?;
    Ok(rx.await?)
}

// =============================================================================
// Backend Implementation
// =============================================================================

/// The Backend actor manages backend operations.
struct Backend {
    // TODO: Add fields as needed
}

impl Backend {
    fn new() -> Self {
        Backend {}
    }
}

// =============================================================================
// Messages
// =============================================================================

/// Messages handled by the Backend actor.
#[derive(Debug)]
enum Message {
    /// Health check request.
    HealthCheck { response: oneshot::Sender<bool> },
}

impl ServiceHandler for Backend {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::HealthCheck { response } => {
                let _ = response.send(true);
            }
        }
    }
}
