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

        // Hard cap on cumulative bytes — fail fast so a malicious sender
        // can't grow `buffer` without bound by streaming chunks. Use
        // saturating_add so a length-overflow here can't sneak past the
        // comparison.
        let after = self.buffer.len().saturating_add(chunk_data.len());
        if after > self.max_total_bytes {
            return ChunkResult::Error(format!(
                "upload exceeds max_upload_mb cap of {} MiB",
                self.max_total_bytes / (1024 * 1024)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn upload(total_chunks: usize, max_total_bytes: usize) -> InFlightUpload {
        InFlightUpload::new(total_chunks, String::new(), false, max_total_bytes)
    }

    #[test]
    fn within_cap_completes() {
        let mut u = upload(2, 1024);
        assert!(matches!(
            u.process_chunk(0, 2, vec![0u8; 400]),
            ChunkResult::InProgress
        ));
        match u.process_chunk(1, 2, vec![0u8; 400]) {
            ChunkResult::Complete { buffer, .. } => assert_eq!(buffer.len(), 800),
            other => panic!(
                "expected Complete, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn first_chunk_over_cap_rejected() {
        let mut u = upload(1, 100);
        match u.process_chunk(0, 1, vec![0u8; 200]) {
            ChunkResult::Error(msg) => assert!(msg.contains("max_upload_mb cap")),
            _ => panic!("expected cap error"),
        }
    }

    #[test]
    fn running_total_breach_mid_stream_rejected() {
        let mut u = upload(3, 1000);
        // 400 + 400 = 800 OK; +300 = 1100 > 1000.
        assert!(matches!(
            u.process_chunk(0, 3, vec![0u8; 400]),
            ChunkResult::InProgress
        ));
        assert!(matches!(
            u.process_chunk(1, 3, vec![0u8; 400]),
            ChunkResult::InProgress
        ));
        match u.process_chunk(2, 3, vec![0u8; 300]) {
            ChunkResult::Error(msg) => assert!(msg.contains("max_upload_mb cap")),
            _ => panic!("expected cap error"),
        }
    }

    #[test]
    fn out_of_order_chunk_rejected() {
        let mut u = upload(2, 1024);
        match u.process_chunk(1, 2, vec![0u8; 10]) {
            ChunkResult::Error(msg) => assert!(msg.contains("Out-of-order")),
            _ => panic!("expected order error"),
        }
    }

    #[test]
    fn chunk_count_mismatch_rejected() {
        let mut u = upload(2, 1024);
        match u.process_chunk(0, 3, vec![0u8; 10]) {
            ChunkResult::Error(msg) => assert!(msg.contains("Chunk count mismatch")),
            _ => panic!("expected count error"),
        }
    }

    #[test]
    fn cap_exactly_at_limit_passes() {
        let mut u = upload(1, 100);
        match u.process_chunk(0, 1, vec![0u8; 100]) {
            ChunkResult::Complete { buffer, .. } => assert_eq!(buffer.len(), 100),
            _ => panic!("expected Complete"),
        }
    }
}
