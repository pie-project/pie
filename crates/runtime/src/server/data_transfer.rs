//! Chunked upload handling.
//!
//! This module provides infrastructure for handling chunked binary uploads,
//! used by both program uploads (`install_program`) and blob transfers
//! (`session.send_file`). The total uploaded byte count is capped per
//! upload via `max_total_bytes` — the running sum is checked on every
//! chunk so a malicious sender can't grow `buffer` without bound.

/// Tracks an in-flight chunked upload (programs or blobs).
pub struct InFlightUpload {
    pub total_chunks: usize,
    pub buffer: Vec<u8>,
    pub next_chunk_index: usize,
    pub manifest: String,
    pub force_overwrite: bool,
    /// Hard cap on the cumulative byte count of all chunks combined.
    /// Comes from `runtime.max_upload_mb` × 1 MiB at server-spawn time.
    pub max_total_bytes: usize,
}

/// Result of processing a chunk in a chunked upload.
pub enum ChunkResult {
    /// Chunk accepted, waiting for more chunks
    InProgress,
    /// All chunks received, upload complete
    Complete {
        buffer: Vec<u8>,
        manifest: String,
        force_overwrite: bool,
    },
    /// Error during chunk processing
    Error(String),
}

impl InFlightUpload {
    /// Creates a new in-flight upload tracker.
    pub fn new(
        total_chunks: usize,
        manifest: String,
        force_overwrite: bool,
        max_total_bytes: usize,
    ) -> Self {
        Self {
            total_chunks,
            buffer: Vec::new(),
            next_chunk_index: 0,
            manifest,
            force_overwrite,
            max_total_bytes,
        }
    }

    /// Process an incoming chunk and return the result.
    ///
    /// Returns `InProgress` if more chunks are expected, `Complete` with the
    /// accumulated buffer when all chunks have been received, or `Error` if
    /// there's a validation failure (out-of-order chunk, count mismatch, or
    /// running total exceeding the configured cap).
    pub fn process_chunk(
        &mut self,
        chunk_index: usize,
        total_chunks: usize,
        mut chunk_data: Vec<u8>,
    ) -> ChunkResult {
        // An upload of no chunks would never satisfy the completion test
        // (`next_chunk_index == total_chunks`, checked after the first
        // chunk), so it would wait forever instead of being rejected.
        if total_chunks == 0 {
            return ChunkResult::Error("Upload must have at least one chunk".to_string());
        }

        if total_chunks != self.total_chunks {
            return ChunkResult::Error(format!(
                "Chunk count mismatch: expected {}, got {}",
                self.total_chunks, total_chunks
            ));
        }

        if chunk_index != self.next_chunk_index {
            return ChunkResult::Error(format!(
                "Out-of-order chunk: expected {}, got {}",
                self.next_chunk_index, chunk_index
            ));
        }

        // saturating_add so a length overflow can't sneak past the cap check.
        let after = self.buffer.len().saturating_add(chunk_data.len());
        if after > self.max_total_bytes {
            return ChunkResult::Error(format!(
                "upload exceeds max_upload_mb cap of {} MiB",
                self.max_total_bytes / (1024 * 1024)
            ));
        }

        self.buffer.append(&mut chunk_data);
        self.next_chunk_index += 1;

        if self.next_chunk_index == self.total_chunks {
            ChunkResult::Complete {
                buffer: std::mem::take(&mut self.buffer),
                manifest: std::mem::take(&mut self.manifest),
                force_overwrite: self.force_overwrite,
            }
        } else {
            ChunkResult::InProgress
        }
    }
}

