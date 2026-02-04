//! Blob upload and download handling.
//!
//! This module handles binary data (blob) transfers between clients and instances,
//! including chunked uploads and downloads.

use std::mem;

use bytes::Bytes;
use pie_client::message::{self, ServerMessage};
use uuid::Uuid;

use crate::instance::InstanceId;
use crate::messaging::{self, PushPullMessage};

use super::session::Session;

/// Tracks an in-flight chunked upload (programs or blobs).
pub struct InFlightUpload {
    pub total_chunks: usize,
    pub buffer: Vec<u8>,
    pub next_chunk_index: usize,
    pub manifest: String,
}

// =============================================================================
// Blob Upload/Download Handlers
// =============================================================================

impl Session {
    pub async fn handle_upload_blob(
        &mut self,
        corr_id: u32,
        instance_id: String,
        blob_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        mut chunk_data: Vec<u8>,
    ) {
        let inst_id = match Uuid::parse_str(&instance_id) {
            Ok(id) => id,
            Err(_) => {
                self.send_response(
                    corr_id,
                    false,
                    format!("Invalid instance_id: {}", instance_id),
                )
                .await;
                return;
            }
        };
        if !self.attached_instances.contains(&inst_id) {
            self.send_response(
                corr_id,
                false,
                format!("Instance not owned by client: {}", instance_id),
            )
            .await;
            return;
        }

        if !self.inflight_blob_uploads.contains_key(&blob_hash) {
            if chunk_index != 0 {
                self.send_response(corr_id, false, "First chunk index must be 0".to_string())
                    .await;
                return;
            }
            self.inflight_blob_uploads.insert(
                blob_hash.clone(),
                InFlightUpload {
                    total_chunks,
                    buffer: Vec::with_capacity(total_chunks * message::CHUNK_SIZE_BYTES),
                    next_chunk_index: 0,
                    manifest: String::new(),
                },
            );
        }

        if let Some(mut inflight) = self.inflight_blob_uploads.get_mut(&blob_hash) {
            if total_chunks != inflight.total_chunks || chunk_index != inflight.next_chunk_index {
                let error_msg = if total_chunks != inflight.total_chunks {
                    format!(
                        "Chunk count mismatch: expected {}, got {}",
                        inflight.total_chunks, total_chunks
                    )
                } else {
                    format!(
                        "Out-of-order chunk: expected {}, got {}",
                        inflight.next_chunk_index, chunk_index
                    )
                };
                self.send_response(corr_id, false, error_msg).await;
                self.inflight_blob_uploads.remove(&blob_hash);
                return;
            }

            inflight.buffer.append(&mut chunk_data);
            inflight.next_chunk_index += 1;

            if inflight.next_chunk_index == total_chunks {
                let final_hash = blake3::hash(&inflight.buffer).to_hex().to_string();

                if final_hash == blob_hash {
                    messaging::pushpull_send(PushPullMessage::PushBlob {
                        topic: inst_id.to_string(),
                        message: Bytes::from(mem::take(&mut inflight.buffer)),
                    })
                    .unwrap();
                    self.send_response(corr_id, true, "Blob sent to instance".to_string())
                        .await;
                } else {
                    self.send_response(
                        corr_id,
                        false,
                        format!("Hash mismatch: expected {}, got {}", blob_hash, final_hash),
                    )
                    .await;
                }
                self.inflight_blob_uploads.remove(&blob_hash);
            }
        }
    }

    pub async fn handle_send_blob(&mut self, inst_id: InstanceId, data: Bytes) {
        let blob_hash = blake3::hash(&data).to_hex().to_string();
        let total_chunks = (data.len() + message::CHUNK_SIZE_BYTES - 1) / message::CHUNK_SIZE_BYTES;

        for (i, chunk) in data.chunks(message::CHUNK_SIZE_BYTES).enumerate() {
            self.send(ServerMessage::DownloadBlob {
                corr_id: 0,
                instance_id: inst_id.to_string(),
                blob_hash: blob_hash.clone(),
                chunk_index: i,
                total_chunks,
                chunk_data: chunk.to_vec(),
            })
            .await;
        }
    }
}
