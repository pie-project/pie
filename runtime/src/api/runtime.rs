//! pie:core/runtime - Runtime information + child-inferlet launch.

use crate::api::pie;
use crate::instance::InstanceState;
use crate::metadata_store;
use crate::model;
use crate::process::{self, ProcessId};
use crate::program::{self, ProgramName};

use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

/// Handle to a launched child inferlet.
///
/// Wraps the oneshot the child's result_tx writes into. The first time the
/// pollable fires it drains the receiver; subsequent calls to `get()` return
/// the cached result.
///
/// Drop = detach: the child runs to completion on its own. To kill the child,
/// call `cancel()` explicitly before dropping.
#[derive(Debug)]
pub struct Child {
    pid: ProcessId,
    receiver: oneshot::Receiver<std::result::Result<String, String>>,
    result: Option<std::result::Result<String, String>>,
    done: bool,
}

#[async_trait]
impl Pollable for Child {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        // Match the FutureString pattern: borrow the receiver (which is
        // Unpin) and await it in place. On RecvError (sender dropped) we
        // store an Err so `get()` always has something to return after
        // `done`.
        self.result = Some(
            (&mut self.receiver)
                .await
                .unwrap_or_else(|_| Err("child sender dropped".to_string())),
        );
        self.done = true;
    }
}

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

    async fn models(&mut self) -> Result<Vec<String>> {
        Ok(model::models())
    }

    async fn max_output_tokens(&mut self) -> Result<u32> {
        Ok(model::min_output_token_ceiling())
    }

    async fn metadata_put(
        &mut self,
        namespace: String,
        key: String,
        value: Vec<u8>,
    ) -> Result<Result<(), String>> {
        let owner = self.metadata_owner();
        Ok(metadata_store::put(&owner, &namespace, &key, value).map_err(|e| e.to_string()))
    }

    async fn metadata_get(
        &mut self,
        namespace: String,
        key: String,
    ) -> Result<Result<Option<Vec<u8>>, String>> {
        let owner = self.metadata_owner();
        Ok(metadata_store::get(&owner, &namespace, &key).map_err(|e| e.to_string()))
    }

    async fn metadata_delete(
        &mut self,
        namespace: String,
        key: String,
    ) -> Result<Result<bool, String>> {
        let owner = self.metadata_owner();
        Ok(metadata_store::delete(&owner, &namespace, &key).map_err(|e| e.to_string()))
    }

    async fn launch(
        &mut self,
        program: String,
        input: String,
    ) -> Result<std::result::Result<Resource<Child>, String>> {
        let username = self.get_username();

        let program_name = match ProgramName::parse(&program) {
            Ok(p) => p,
            Err(e) => return Ok(Err(format!("invalid program name: {e}"))),
        };

        if let Err(e) = program::install(&program_name).await {
            return Ok(Err(format!("install failed: {e}")));
        }

        let (tx, rx) = oneshot::channel();
        let pid = match process::spawn(
            username,
            program_name,
            input,
            None,  // client_id: detached
            false, // capture_outputs
            Some(tx),
            None, // workflow_id
            None, // token_budget: default
        ) {
            Ok(pid) => pid,
            Err(e) => return Ok(Err(format!("spawn failed: {e}"))),
        };

        let child = Child {
            pid,
            receiver: rx,
            result: None,
            done: false,
        };
        Ok(Ok(self.ctx().table.push(child)?))
    }
}

impl pie::core::runtime::HostChild for InstanceState {
    async fn pollable(&mut self, this: Resource<Child>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(
        &mut self,
        this: Resource<Child>,
    ) -> Result<Option<std::result::Result<String, String>>> {
        let child = self.ctx().table.get(&this)?;
        if child.done {
            Ok(child.result.clone())
        } else {
            Ok(None)
        }
    }

    async fn pid(&mut self, this: Resource<Child>) -> Result<String> {
        let child = self.ctx().table.get(&this)?;
        Ok(child.pid.to_string())
    }

    async fn cancel(&mut self, this: Resource<Child>) -> Result<()> {
        let child = self.ctx().table.get(&this)?;
        // Fire-and-forget terminate. If the child already finished, the
        // process actor has exited and this send is a no-op.
        process::terminate(child.pid, Err("cancelled".to_string()));
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Child>) -> Result<()> {
        // Drop = detach. The child keeps running; we just discard the
        // oneshot receiver. The child's result_tx.send() will fail silently
        // when the child finishes, but that's a no-op for the process actor.
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
