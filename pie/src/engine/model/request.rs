use anyhow::{Result, bail};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::oneshot;

pub static HANDSHAKE_ID: u32 = 0;
pub static HEARTBEAT_ID: u32 = 1;
pub static QUERY_ID: u32 = 2;
pub static FORWARD_PASS_ID: u32 = 3;
pub static EMBED_IMAGE_ID: u32 = 4;
pub static INITIALIZE_ADAPTER_ID: u32 = 5;
pub static UPDATE_ADAPTER_ID: u32 = 6;
pub static UPLOAD_ADAPTER_ID: u32 = 7;
pub static DOWNLOAD_ADAPTER_ID: u32 = 8;

#[derive(Debug)]
pub enum Request {
    Handshake(HandshakeRequest, oneshot::Sender<HandshakeResponse>),
    Query(QueryRequest, oneshot::Sender<QueryResponse>),
    Synchronize(oneshot::Sender<()>),
    Heartbeat(HeartbeatRequest),
    ForwardPass(
        ForwardPassRequest,
        Option<oneshot::Sender<ForwardPassResponse>>,
    ),
    EmbedImage(EmbedImageRequest),
    InitializeAdapter(InitializeAdapterRequest),
    UpdateAdapter(UpdateAdapterRequest),
    UploadAdapter(UploadAdapterRequest),
    DownloadAdapter(DownloadAdapterRequest, oneshot::Sender<Bytes>),
}

impl Request {
    pub fn is_eager(&self) -> bool {
        match self {
            Request::ForwardPass(_, _) => false,
            _ => true,
        }
    }

    pub fn is_sync_req(&self) -> bool {
        match self {
            Request::Synchronize(_) => true,
            _ => false,
        }
    }

    pub fn has_response(&self) -> bool {
        match self {
            Request::Handshake(_, _) => true,
            Request::Query(_, _) => true,
            Request::ForwardPass(_, r) => r.is_some(),
            Request::DownloadAdapter(_, _) => true,
            _ => false,
        }
    }

    pub fn handler_id(&self) -> u32 {
        match self {
            Request::Handshake(_, _) => HANDSHAKE_ID,
            Request::Query(_, _) => QUERY_ID,
            Request::Synchronize(_) => HEARTBEAT_ID,
            Request::Heartbeat(_) => HEARTBEAT_ID,
            Request::ForwardPass(_, _) => FORWARD_PASS_ID,
            Request::EmbedImage(_) => EMBED_IMAGE_ID,
            Request::InitializeAdapter(_) => INITIALIZE_ADAPTER_ID,
            Request::UpdateAdapter(_) => UPDATE_ADAPTER_ID,
            Request::UploadAdapter(_) => UPLOAD_ADAPTER_ID,
            Request::DownloadAdapter(_, _) => DOWNLOAD_ADAPTER_ID,
        }
    }

    pub fn serialize_req(&self) -> Result<Bytes> {
        let b = match self {
            Request::Handshake(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::Query(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::Synchronize(_) => bail!("cannot serialize synchronize request"),
            Request::Heartbeat(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::ForwardPass(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::EmbedImage(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::InitializeAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::UpdateAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::UploadAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::DownloadAdapter(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
        };
        Ok(b)
    }

    pub fn deserialize_resp(self, b: Bytes) -> Result<()> {
        match self {
            Request::Handshake(_, resp) => {
                let r: HandshakeResponse = rmp_serde::from_slice(&b)?;
                resp.send(r).ok();
            }
            Request::Query(_, resp) => {
                let r: QueryResponse = rmp_serde::from_slice(&b)?;
                resp.send(r).ok();
            }
            Request::ForwardPass(_, resp) => {
                let r: ForwardPassResponse = rmp_serde::from_slice(&b)?;
                if let Some(tx) = resp {
                    tx.send(r).ok();
                }
            }
            Request::DownloadAdapter(_, resp) => {
                resp.send(b).ok();
            }
            _ => {
                bail!("cannot deserialize response for request {:?}", self);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub version: String,
    pub model_name: String,
    pub model_traits: Vec<String>,
    pub model_description: String,
    pub prompt_template: String,
    pub prompt_template_type: String,
    pub prompt_stop_tokens: Vec<String>,
    pub kv_page_size: u32,
    pub max_batch_tokens: usize,
    pub resources: HashMap<u32, u32>,
    pub tokenizer_num_vocab: usize,
    pub tokenizer_merge_table: HashMap<u32, Vec<u8>>,
    pub tokenizer_special_tokens: HashMap<String, u32>,
    pub tokenizer_split_regex: String,
    pub tokenizer_escape_non_printable: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub value: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HeartbeatRequest {}

#[derive(Debug, Serialize, Deserialize)]
pub struct HeartbeatResponse {}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassRequest {
    pub input_tokens: Vec<u32>,
    pub input_token_positions: Vec<u32>,
    pub input_embed_ptrs: Vec<u32>,
    pub input_embed_positions: Vec<u32>,
    pub adapter: Option<u32>,
    pub adapter_seed: Option<i64>,
    pub mask: Vec<Vec<u32>>,
    pub kv_page_ptrs: Vec<u32>,
    pub kv_page_last_len: u32,
    pub output_token_indices: Vec<u32>,
    pub output_token_samplers: Vec<HashMap<String, rmpv::Value>>,
    pub output_embed_ptrs: Vec<u32>,
    pub output_embed_indices: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    pub tokens: Vec<u32>,
    pub dists: Vec<(Vec<u32>, Vec<f32>)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedImageRequest {
    pub embed_ptrs: Vec<u32>,
    pub image_blob: Vec<u8>,
    pub position_offset: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeAdapterRequest {
    pub adapter_ptr: u32,
    pub rank: u32,
    pub alpha: f32,
    pub population_size: u32,
    pub mu_fraction: f32,
    pub initial_sigma: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateAdapterRequest {
    pub adapter_ptr: u32,
    pub scores: Vec<f32>,
    pub seeds: Vec<i64>,
    pub max_sigma: f32,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct UploadAdapterRequest {
    pub adapter_ptr: u32,
    pub name: String,
    pub adapter_data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DownloadAdapterRequest {
    pub adapter_ptr: u32,
    pub name: String,
}
