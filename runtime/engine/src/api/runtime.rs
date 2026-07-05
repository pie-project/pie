//! pie:core/runtime - Runtime information + host async timer.

use crate::api::pie;
use crate::instance::InstanceState;

use anyhow::Result;
use std::time::Duration;
use wasmtime::component::{Accessor, HasSelf};

impl pie::core::runtime::Host for InstanceState {
    async fn version(&mut self) -> Result<String> {
        Ok(env!("CARGO_PKG_VERSION").to_string())
    }

    async fn instance_id(&mut self) -> Result<String> {
        Ok(self.id().to_string())
    }

    async fn username(&mut self) -> Result<String> {
        Ok(self.get_username())
    }
}

impl pie::core::runtime::HostWithStore<InstanceState> for HasSelf<InstanceState> {
    // Host async timer: the wasmtime-46 event loop drives the await, so the
    // guest task suspends without any guest-side executor.
    async fn sleep(_accessor: &Accessor<InstanceState, Self>, duration_ns: u64) -> Result<()> {
        tokio::time::sleep(Duration::from_nanos(duration_ns)).await;
        Ok(())
    }
}
