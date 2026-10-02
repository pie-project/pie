pub struct InFlightUpload {
    pub total_chunks: usize,
    pub buffer: Vec<u8>,
    pub next_chunk_index: usize,
    pub file: String,
    pub version: Option<String>,
    pub force_overwrite: bool,
    pub max_total_bytes: usize,
}

pub enum ChunkResult {
    InProgress,
    Complete {
        buffer: Vec<u8>,
        file: String,
        version: Option<String>,
        force_overwrite: bool,
    },
    Error(String),
}

impl InFlightUpload {
    pub fn new(
        total_chunks: usize,
        file: String,
        version: Option<String>,
        force_overwrite: bool,
        max_total_bytes: usize,
    ) -> Self {
        Self {
            total_chunks,
            buffer: Vec::new(),
            next_chunk_index: 0,
            file,
            version,
            force_overwrite,
            max_total_bytes,
        }
    }

    pub fn process_chunk(
        &mut self,
        chunk_index: usize,
        total_chunks: usize,
        mut chunk_data: Vec<u8>,
    ) -> ChunkResult {
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
                file: std::mem::take(&mut self.file),
                version: self.version.take(),
                force_overwrite: self.force_overwrite,
            }
        } else {
            ChunkResult::InProgress
        }
    }
}
