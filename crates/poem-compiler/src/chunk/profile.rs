//! What a device tells the chunk plan: how its backend grows memory, what one
//! kernel binding may span, how offsets align, and the load it expects.

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// The device's inputs to [`super::plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkProfile {
    /// The chunk size the plan aims near; it picks among multiples of the
    /// list slot from half to twice this.
    pub target_chunk_bytes: u64,
    /// The bytes the backend grows or shrinks the pool by at a time (a VMM
    /// or sparse-heap segment); a chunk never straddles one, so a segment
    /// holds the whole chunks that fit, rounded up to `segment_grain`.
    pub segment_bytes: u64,
    /// The granularity a segment is mapped in (CUDA VMM: 2 MiB; a Metal
    /// sparse heap: 16 KiB pages).
    pub segment_grain: u64,
    /// The most bytes one buffer binding a kernel reads may span (wgpu:
    /// 128 MiB by default); a chunk is never larger.
    pub max_binding_bytes: u64,
    /// Where every row inside a slot starts, and every slot's size, aligned.
    pub row_align: u64,
    /// The tile a tiled realization lays rows in (TPU `[rows, 128]` u32:
    /// 8 x 128 x 4 B); 0 where the backend asks none. Rows and slots then
    /// align to it too.
    pub tile_bytes: u64,
    /// The working sets the device expects resident at once.
    pub concurrency: u32,
    /// The context length a working set is expected to reach, in tokens.
    pub context: u32,
    /// Whether a routed expert larger than the chunk is cut into pieces
    /// along its banks' output axis ([`super::weights`]); otherwise the
    /// chunk is never smaller than an expert.
    pub cut_experts: bool,
}

impl ChunkProfile {
    /// A server CUDA card: 16 MiB chunks out of 1 GiB VMM segments, a
    /// hundred-odd working sets at 32k tokens.
    pub const SERVER_CUDA: ChunkProfile = ChunkProfile {
        target_chunk_bytes: 16 * MIB,
        segment_bytes: GIB,
        segment_grain: 2 * MIB,
        max_binding_bytes: 4 * GIB,
        row_align: 256,
        tile_bytes: 0,
        concurrency: 128,
        context: 32 * 1024,
        cut_experts: false,
    };

    /// An on-device Metal GPU: chunks of a couple of MiB out of 64 MiB
    /// sparse-heap segments, a handful of working sets at 8k tokens.
    pub const ON_DEVICE_METAL: ChunkProfile = ChunkProfile {
        target_chunk_bytes: 2 * MIB,
        segment_bytes: 64 * MIB,
        segment_grain: 16 * KIB,
        max_binding_bytes: 256 * MIB,
        row_align: 256,
        tile_bytes: 0,
        concurrency: 4,
        context: 8 * 1024,
        cut_experts: true,
    };

    /// A TPU pool fixed at load, realized as `[chunks, rows, 128]` u32 tiles.
    pub const TPU: ChunkProfile = ChunkProfile {
        target_chunk_bytes: 16 * MIB,
        segment_bytes: 16 * GIB,
        segment_grain: 4 * KIB,
        max_binding_bytes: 16 * GIB,
        row_align: 256,
        tile_bytes: 8 * 128 * 4,
        concurrency: 128,
        context: 32 * 1024,
        cut_experts: false,
    };

    /// What every row offset and slot size aligns to.
    #[must_use]
    pub fn align(&self) -> u64 {
        self.row_align.max(self.tile_bytes).max(1)
    }

    /// The largest chunk a backend could map and bind at all.
    #[must_use]
    pub fn most(&self) -> u64 {
        self.max_binding_bytes.min(self.segment_bytes)
    }

    /// The bytes a segment of whole `chunk`-byte chunks maps, and the
    /// chunks it holds.
    #[must_use]
    pub fn segment(&self, chunk: u64) -> (u64, u64) {
        let chunks = (self.segment_bytes / chunk.max(1)).max(1);
        let grain = self.segment_grain.max(1);
        ((chunks * chunk).div_ceil(grain) * grain, chunks)
    }

    /// The chunk sizes the plan may pick from: half to twice the target,
    /// never past a binding or a segment.
    #[must_use]
    pub fn range(&self) -> (u64, u64) {
        let hi = (2 * self.target_chunk_bytes)
            .min(self.max_binding_bytes)
            .min(self.segment_bytes);
        ((self.target_chunk_bytes / 2).min(hi).max(KIB), hi.max(KIB))
    }
}

impl Default for ChunkProfile {
    fn default() -> ChunkProfile {
        ChunkProfile::SERVER_CUDA
    }
}
