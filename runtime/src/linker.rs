//! Linker service.
//!
//! Singleton service that owns the pre-configured wasmtime Engine.
//! Creates per-instance linkers with WASI, WASI HTTP, Pie API host bindings,
//! and dynamically linked library dependencies.

mod dynamic_linking;
mod state;

use std::sync::LazyLock;

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use wasmtime::component::{Component, Instance as WasmInstance, Linker as WasmLinker};
use wasmtime::{Engine, Store};

use crate::api;
use crate::program::{self, ProgramName};
use crate::runtime::instance::InstanceState;
use crate::runtime::output::OutputDeliveryCtrl;
use crate::runtime::instance::InstanceId;
use crate::service::{Service, ServiceHandler};

// ---- Singleton Actor --------------------------------------------------------

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the linker service with the given engine.
pub fn spawn(engine: &Engine) {
    SERVICE
        .spawn(|| Linker::new(engine))
        .expect("linker already spawned");
}

// ---- Public API (message wrappers) ------------------------------------------


/// Link and instantiate a program with its dependencies.
pub async fn instantiate(
    inst_id: InstanceId,
    username: String,
    program_name: &ProgramName,
) -> Result<(Store<InstanceState>, WasmInstance)> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Instantiate {
        inst_id,
        username,
        program_name: program_name.clone(),
        response: tx,
    })?;
    rx.await?
}

// ---- State ------------------------------------------------------------------

struct Linker {
    engine: Engine,
}

impl Linker {
    fn new(engine: &Engine) -> Self {
        Linker { engine: engine.clone() }
    }

    async fn instantiate(
        &self,
        inst_id: InstanceId,
        username: String,
        program_name: &ProgramName,
    ) -> Result<(Store<InstanceState>, WasmInstance)> {
        // 1. Get the main component
        let component = program::get_wasm_component(program_name)
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;

        // 2. Get dependency components
        let dependency_components = self.resolve_dependency_components(program_name).await?;

        // 3. Create instance state and store
        let (inst_state, output_ctrl) = InstanceState::new(inst_id, username).await;
        let mut store = Store::new(&self.engine, inst_state);

        // 4. Create and configure linker
        let mut linker = WasmLinker::<InstanceState>::new(&self.engine);

        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .expect("Failed to link WASI");
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
            .expect("Failed to link WASI HTTP");

        api::add_to_linker(&mut linker)?;

        // 5. Instantiate library dependencies (dynamic linking)
        if !dependency_components.is_empty() {
            dynamic_linking::instantiate_libraries(
                &self.engine,
                &mut linker,
                &mut store,
                dependency_components,
            )
            .await?;
        }

        // 6. Instantiate the main component
        let instance = linker
            .instantiate_async(&mut store, &component)
            .await
            .map_err(|e| anyhow!("Instantiation error: {e}"))?;

        Ok((store, instance))
    }

    /// Resolve and fetch all dependency components for a program.
    async fn resolve_dependency_components(
        &self,
        program_name: &ProgramName,
    ) -> Result<Vec<Component>> {
        let manifest = program::fetch_manifest(program_name)
            .await
            .ok_or_else(|| anyhow!("Manifest not found for: {}", program_name))?;

        let dep_names = manifest.dependency_names();
        let mut components = Vec::with_capacity(dep_names.len());

        for dep_name in dep_names {
            let component = program::get_wasm_component(&dep_name)
                .await
                .ok_or_else(|| anyhow!("Dependency component not found: {}", dep_name))?;
            components.push(component);
        }

        Ok(components)
    }
}

// ---- Messages ---------------------------------------------------------------

enum Message {
    Instantiate {
        inst_id: InstanceId,
        username: String,
        program_name: ProgramName,
        response: oneshot::Sender<Result<(Store<InstanceState>, WasmInstance)>>,
    },
}

impl ServiceHandler for Linker {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Instantiate { inst_id, username, program_name, response } => {
                let _ = response.send(
                    self.instantiate(inst_id, username, &program_name).await
                );
            }
        }
    }
}
