pub mod forward;
pub mod kvs;
pub mod message;
pub mod runtime;
pub mod tokenize;

use crate::api::inferlet;
use crate::api::inferlet::core::common::Priority;
use crate::instance::InstanceState;
use crate::model;
use crate::model::request::{FormatChatRequest, FormatChatResponse, QueryRequest, QueryResponse, Request};
use crate::model::resource::{ResourceId, ResourceTypeId};
use crate::model::{ModelInfo, submit_request};
use anyhow::Result;
use bytes::Bytes;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};
use wasmtime_wasi::{WasiView, async_trait};

// A counter to generate unique stream IDs for new queues
static NEXT_QUEUE_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Debug, Clone)]
pub struct Model {
    pub service_id: usize,
    pub info: Arc<ModelInfo>,
}

#[derive(Debug, Clone)]
pub struct Blob {
    pub data: Bytes,
}

#[derive(Debug, Clone)]
pub struct Queue {
    pub service_id: usize,
    pub info: Arc<ModelInfo>,
    pub uid: u32,
    pub priority: u32,
}

#[derive(Debug)]
pub struct DebugQueryResult {
    receiver: oneshot::Receiver<QueryResponse>,
    result: Option<String>,
    done: bool,
}

#[derive(Debug)]
pub struct FormatChatResult {
    receiver: Option<oneshot::Receiver<FormatChatResponse>>,
    result: Option<Vec<u32>>,
    done: bool,
}

#[derive(Debug)]
pub struct SynchronizationResult {
    receiver: oneshot::Receiver<()>,
    done: bool,
}

#[derive(Debug)]
pub struct BlobResult {
    pub(crate) receiver: oneshot::Receiver<Bytes>,
    pub(crate) result: Option<Bytes>,
    pub(crate) done: bool,
}

#[async_trait]
impl Pollable for DebugQueryResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let res = (&mut self.receiver).await.unwrap();
        let res_string = res.value;
        self.result = Some(res_string);
        self.done = true;
    }
}

#[async_trait]
impl Pollable for FormatChatResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let receiver = self.receiver.as_mut().expect("FormatChatResult: no receiver and not done");
        let res = receiver.await.unwrap();
        self.result = Some(res.token_ids);
        self.done = true;
    }
}

#[async_trait]
impl Pollable for BlobResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let res = (&mut self.receiver).await.unwrap();
        self.result = Some(res);

        self.done = true;
    }
}

#[async_trait]
impl Pollable for SynchronizationResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let _ = (&mut self.receiver).await.unwrap();
        self.done = true;
    }
}

impl inferlet::core::common::Host for InstanceState {
    async fn allocate_resources(
        &mut self,
        queue: Resource<Queue>,
        resource_type: ResourceTypeId,
        count: u32,
    ) -> Result<Vec<ResourceId>> {
        let inst_id = self.id();
        let svc_id = self.ctx().table.get(&queue)?.service_id;
        let (tx, rx) = oneshot::channel();

        model::Command::Allocate {
            inst_id,
            type_id: resource_type,
            count: count as usize,
            response: tx,
        }
        .dispatch(svc_id)?;

        let phys_ptrs = rx.await?;
        let virt_ptrs = self.map_resources(svc_id, resource_type, &phys_ptrs);

        Ok(virt_ptrs)
    }

    async fn deallocate_resources(
        &mut self,
        queue: Resource<Queue>,
        resource_type: ResourceTypeId,
        ptrs: Vec<ResourceId>,
    ) -> Result<()> {
        let inst_id = self.id();
        let svc_id = self.ctx().table.get(&queue)?.service_id;
        self.unmap_resources(svc_id, resource_type, &ptrs);

        model::Command::Deallocate {
            inst_id,
            type_id: resource_type,
            ptrs,
        }
        .dispatch(svc_id)?;

        Ok(())
    }

    async fn get_all_exported_resources(
        &mut self,
        queue: Resource<Queue>,
        resource_type: ResourceTypeId,
    ) -> Result<Vec<(String, u32)>> {
        let q = self.ctx().table.get(&queue)?;
        let (tx, rx) = oneshot::channel();
        model::Command::GetAllExported {
            type_id: resource_type,
            response: tx,
        }
        .dispatch(q.service_id)?;

        // convert list of phys ptrs -> size
        let c = rx
            .await?
            .into_iter()
            .map(|(s, v)| (s, v.len() as u32))
            .collect();

        Ok(c)
    }

    async fn release_exported_resources(
        &mut self,
        queue: Resource<Queue>,
        resource_type: ResourceTypeId,
        name: String,
    ) -> Result<()> {
        let inst_id = self.id();
        let svc_id = self.ctx().table.get(&queue)?.service_id;
        model::Command::ReleaseExported {
            inst_id,
            type_id: resource_type,
            name,
        }
        .dispatch(svc_id)?;

        Ok(())
    }

    async fn export_resources(
        &mut self,
        queue: Resource<Queue>,
        resource_type: ResourceTypeId,
        mut ptrs: Vec<ResourceId>,
        name: String,
    ) -> Result<()> {
        let inst_id = self.id();
        let svc_id = self.ctx().table.get(&queue)?.service_id;

        ptrs.iter_mut().try_for_each(|ptr| {
            *ptr = self.translate_resource_ptr(svc_id, resource_type, *ptr)?;
            Ok::<_, anyhow::Error>(())
        })?;

        model::Command::Export {
            inst_id,
            type_id: resource_type,
            ptrs,
            name,
        }
        .dispatch(svc_id)?;

        Ok(())
    }

    async fn export_resources_sync(
        &mut self,
        queue: Resource<Queue>,
        resource_type: ResourceTypeId,
        mut ptrs: Vec<ResourceId>,
        name: String,
    ) -> Result<Result<(), String>> {
        let inst_id = self.id();
        let svc_id = self.ctx().table.get(&queue)?.service_id;

        ptrs.iter_mut().try_for_each(|ptr| {
            *ptr = self.translate_resource_ptr(svc_id, resource_type, *ptr)?;
            Ok::<_, anyhow::Error>(())
        })?;

        let (tx, rx) = oneshot::channel();
        model::Command::ExportSync {
            inst_id,
            type_id: resource_type,
            ptrs,
            name,
            response: tx,
        }
        .dispatch(svc_id)?;

        Ok(rx.await?)
    }

    async fn import_resources(
        &mut self,
        queue: Resource<Queue>,
        resource_type: ResourceTypeId,
        name: String,
    ) -> Result<Vec<ResourceId>> {
        let inst_id = self.id();
        let svc_id = self.ctx().table.get(&queue)?.service_id;

        let (tx, rx) = oneshot::channel();

        model::Command::Import {
            inst_id,
            type_id: resource_type,
            name,
            response: tx,
        }
        .dispatch(svc_id)?;

        let phys_ptrs = rx.await?;
        let virt_ptrs = self.map_resources(svc_id, resource_type, &phys_ptrs);

        Ok(virt_ptrs)
    }
}

impl inferlet::core::common::HostModel for InstanceState {
    async fn get_name(&mut self, this: Resource<Model>) -> Result<String> {
        let name = self.ctx().table.get(&this)?.info.name.clone();
        Ok(name)
    }
    async fn get_traits(&mut self, this: Resource<Model>) -> Result<Vec<String>> {
        let traits = self.ctx().table.get(&this)?.info.traits.clone();
        Ok(traits)
    }
    async fn get_description(&mut self, this: Resource<Model>) -> Result<String> {
        let description = self.ctx().table.get(&this)?.info.description.clone();
        Ok(description)
    }

    async fn get_prompt_template(&mut self, this: Resource<Model>) -> Result<String> {
        let prompt_template = self.ctx().table.get(&this)?.info.prompt_template.clone();
        Ok(prompt_template)
    }

    async fn get_stop_tokens(&mut self, this: Resource<Model>) -> Result<Vec<String>> {
        let stop_tokens = self.ctx().table.get(&this)?.info.prompt_stop_tokens.clone();
        Ok(stop_tokens)
    }

    async fn get_generation_defaults(
        &mut self,
        this: Resource<Model>,
    ) -> Result<inferlet::core::common::GenerationDefaults> {
        // Sourced from HF `generation_config.json` via the Python worker's
        // handshake response (A5 `extract_generation_defaults`, A6 plumbing).
        // All-None is still a valid outcome for models that ship no
        // generation_config (Mistral, Gemma, Yi, ...); inferlets must
        // fall back to their own neutral defaults.
        let defaults = &self.ctx().table.get(&this)?.info.generation_defaults;
        Ok(inferlet::core::common::GenerationDefaults {
            temperature: defaults.temperature,
            top_p: defaults.top_p,
            top_k: defaults.top_k,
            min_p: defaults.min_p,
            repetition_penalty: defaults.repetition_penalty,
        })
    }

    async fn get_service_id(&mut self, this: Resource<Model>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.service_id as u32)
    }

    async fn get_kv_page_size(&mut self, this: Resource<Model>) -> Result<u32> {
        let kv_page_size = self.ctx().table.get(&this)?.info.kv_page_size;
        Ok(kv_page_size)
    }

    async fn create_queue(&mut self, this: Resource<Model>) -> Result<Resource<Queue>> {
        let model = self.ctx().table.get(&this)?;
        let queue = Queue {
            service_id: model.service_id,
            info: model.info.clone(),
            uid: NEXT_QUEUE_ID.fetch_add(1, Ordering::SeqCst),
            priority: 0,
        };
        let res = self.ctx().table.push(queue)?;
        Ok(res)
    }

    async fn drop(&mut self, this: Resource<Model>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::core::common::HostQueue for InstanceState {
    async fn get_service_id(&mut self, this: Resource<Queue>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.service_id as u32)
    }

    async fn synchronize(
        &mut self,
        this: Resource<Queue>,
    ) -> Result<Resource<SynchronizationResult>> {
        let (svc_id, queue_id, priority) = self.read_queue(&this)?;
        let (tx, rx) = oneshot::channel();
        let req = Request::Synchronize(tx);

        submit_request(svc_id, queue_id, priority, req)?;

        let result = SynchronizationResult {
            receiver: rx,
            done: false,
        };
        Ok(self.ctx().table.push(result)?)
    }

    async fn set_priority(&mut self, this: Resource<Queue>, priority: Priority) -> Result<()> {
        let queue = self.ctx().table.get_mut(&this)?;

        queue.priority = match priority {
            Priority::Low => 0,
            Priority::Normal => 0,
            Priority::High => 1,
        };
        Ok(())
    }

    async fn debug_query(
        &mut self,
        this: Resource<Queue>,
        query: String,
    ) -> Result<Resource<DebugQueryResult>> {
        let (svc_id, queue_id, priority) = self.read_queue(&this)?;
        let (tx, rx) = oneshot::channel();
        let req = Request::Query(QueryRequest { query }, tx);

        submit_request(svc_id, queue_id, priority, req)?;

        let res = DebugQueryResult {
            receiver: rx,
            result: None,
            done: false,
        };

        Ok(self.ctx().table.push(res)?)
    }

    async fn format_chat(
        &mut self,
        this: Resource<Queue>,
        messages_json: String,
        tools_json: Option<String>,
        add_generation_prompt: bool,
    ) -> Result<Resource<FormatChatResult>> {
        let queue = self.ctx().table.get(&this)?;
        let info = queue.info.clone();
        let svc_id = queue.service_id;
        let queue_id = queue.uid;
        let priority = queue.priority;

        // Fast path: render + tokenize in-process (validated at startup).
        if info.use_minijinja_fast_path {
            let t_fc = std::time::Instant::now();
            match model::chat_template::render_and_tokenize(
                &info.chat_template,
                &messages_json,
                tools_json.as_deref(),
                add_generation_prompt,
                &info.tokenizer,
            ) {
                Ok(token_ids) => {
                    if std::env::var("PIE_TTFT_TRACE").is_ok() {
                        eprintln!(
                            "[FORMAT-CHAT-TRACE] wit_handler_total={:.2}ms tokens={}",
                            t_fc.elapsed().as_secs_f64() * 1000.0,
                            token_ids.len(),
                        );
                    }
                    let res = FormatChatResult {
                        receiver: None,
                        result: Some(token_ids),
                        done: true,
                    };
                    return Ok(self.ctx().table.push(res)?);
                }
                Err(reason) => {
                    eprintln!("[WARN] minijinja fast path failed, falling back to RPC: {}", reason);
                }
            }
        }

        // Fallback: Python RPC path.
        let (tx, rx) = oneshot::channel();
        let req = Request::FormatChat(
            FormatChatRequest { messages_json, tools_json, add_generation_prompt },
            tx,
        );
        submit_request(svc_id, queue_id, priority, req)?;

        let res = FormatChatResult {
            receiver: Some(rx),
            result: None,
            done: false,
        };
        Ok(self.ctx().table.push(res)?)
    }

    async fn drop(&mut self, this: Resource<Queue>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::core::common::HostDebugQueryResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<DebugQueryResult>,
    ) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<DebugQueryResult>) -> Result<Option<String>> {
        let result = self.ctx().table.get_mut(&this)?;
        if result.done {
            Ok(result.result.clone())
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<DebugQueryResult>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::core::common::HostFormatChatResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<FormatChatResult>,
    ) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<FormatChatResult>) -> Result<Option<Vec<u32>>> {
        let result = self.ctx().table.get_mut(&this)?;
        if result.done {
            Ok(result.result.take())
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<FormatChatResult>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::core::common::HostSynchronizationResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<SynchronizationResult>,
    ) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<SynchronizationResult>) -> Result<Option<bool>> {
        let result = self.ctx().table.get_mut(&this)?;
        if result.done {
            Ok(Some(true))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<SynchronizationResult>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::core::common::HostBlob for InstanceState {
    async fn new(&mut self, data: Vec<u8>) -> Result<Resource<Blob>> {
        let blob = Blob {
            data: Bytes::from(data),
        };
        let res = self.ctx().table.push(blob)?;
        Ok(res)
    }

    async fn read(&mut self, this: Resource<Blob>, offset: u64, size: u64) -> Result<Vec<u8>> {
        let blob = self.ctx().table.get(&this)?;
        let data = blob
            .data
            .get(offset as usize..(offset + size) as usize)
            .unwrap();
        Ok(data.to_vec())
    }

    async fn size(&mut self, this: Resource<Blob>) -> Result<u64> {
        let blob = self.ctx().table.get(&this)?;
        Ok(blob.data.len() as u64)
    }

    async fn drop(&mut self, this: Resource<Blob>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::core::common::HostBlobResult for InstanceState {
    async fn pollable(&mut self, this: Resource<BlobResult>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<BlobResult>) -> Result<Option<Resource<Blob>>> {
        let has_result = self.ctx().table.get(&this)?.result.is_some();
        if has_result {
            let data = mem::take(&mut self.ctx().table.get_mut(&this)?.result).unwrap();
            let blob = Blob { data };
            Ok(Some(self.ctx().table.push(blob)?))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<BlobResult>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
