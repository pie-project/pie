//! Chunked upload handling.
//!
//! This module provides infrastructure for handling chunked binary uploads,
//! used by both program uploads and blob transfers.

/// Maximum allowed size for a single chunk (256MB safety limit)
const MAX_CHUNK_SIZE: usize = 256 * 1024 * 1024;

/// Tracks an in-flight chunked upload (programs or blobs).
pub struct InFlightUpload {
    pub total_chunks: usize,
    pub buffer: Vec<u8>,
    pub next_chunk_index: usize,
    pub manifest: String,
    pub force_overwrite: bool,
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
    pub fn new(total_chunks: usize, manifest: String, force_overwrite: bool) -> Self {
        Self {
            total_chunks,
            buffer: Vec::new(),
            next_chunk_index: 0,
            manifest,
            force_overwrite,
        }
    }

    /// Process an incoming chunk and return the result.
    ///
    /// Returns `InProgress` if more chunks are expected, `Complete` with the
    /// accumulated buffer when all chunks have been received, or `Error` if
    /// there's a validation failure.
    pub fn process_chunk(
        &mut self,
        chunk_index: usize,
        total_chunks: usize,
        mut chunk_data: Vec<u8>,
    ) -> ChunkResult {
        // Safety limit: reject absurdly large chunks
        if chunk_data.len() > MAX_CHUNK_SIZE {
            return ChunkResult::Error(format!(
                "Chunk size {} exceeds maximum allowed size",
                chunk_data.len()
            ));
        }

        // Validate chunk consistency
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

        // Accumulate chunk
        self.buffer.append(&mut chunk_data);
        self.next_chunk_index += 1;

        // Check if complete
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
