//! The API half a backend crate supplies: its device, its buffers, its
//! frames. Everything else in this crate is generic over [`Backend`] and is
//! monomorphized per backend, so a call through these traits is static.

use crate::error::{Fault, Result};

/// Where a reservation lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    Device,

    Host,

    Staging,

    /// Host memory the device writes and the host maps to read (WebGPU's
    /// `MAP_READ`); a backend with no such kind serves it as [`Memory::Host`].
    Readback,
}

/// What runs when a submitted frame lands: `Some(why)` if it faulted.
pub type OnDone = Box<dyn FnOnce(Option<String>) + Send + 'static>;

/// A GPU API: the types a backend crate binds the shared engine to.
pub trait Backend:
    Sized + Copy + Default + std::fmt::Debug + Eq + std::hash::Hash + Send + Sync + 'static
{
    /// The API's name in a message.
    const NAME: &'static str;

    /// Whether a host buffer is mapped into this process: the host lays and
    /// reads it in place ([`Buffer::mapped_ptr`]) and frames copy between it
    /// and the device. Else the host reaches device memory only through the
    /// queue, and host-side copies live in process memory.
    const MAPS_HOST: bool;

    /// The bytes a buffer-to-buffer copy's offsets and length are a
    /// multiple of.
    const COPY_GRAIN: u64;

    /// The storage target a checkpoint lands for.
    const STORAGE: checkpoint::types::BackendKind;

    /// The device one load binds.
    type Context: Context<Self>;

    /// One device reservation; clones share it.
    type Buffer: Buffer<Self>;

    /// Device work recorded and not yet submitted.
    type Frame: Frame<Self>;

    /// A submitted frame, until it lands.
    type Pending: Pending;

    /// The device's tuning profile, as the dispatch reads it.
    type Tuning: Tactics;

    /// Whether `domain` names this API's device.
    fn is_device(domain: engine::transfer::MemoryDomain) -> bool;

    /// Runs `f` on the frame this thread records into, if one is.
    fn with_frame<R>(f: impl FnOnce(&Self::Frame) -> R) -> Option<R> {
        crate::probe::with_frame::<Self::Frame, R>(f)
    }

    /// Submits what this thread's recording frame holds, if one is.
    fn flush_recording() -> Result<()> {
        Self::with_frame(Frame::flush).unwrap_or(Ok(()))
    }

    /// Lays `len` bytes at `at` of `staging` by `fill`, handed the address
    /// of the first, before work the frame records next reads them.
    fn stage_rows(
        staging: &Self::Buffer,
        at: u64,
        len: u64,
        fill: impl FnOnce(usize) -> Result<()>,
    ) -> Result<()>;

    /// Records a copy into `frame` whose ends the frame does not keep alive:
    /// their owner outlives every frame.
    fn copy_unretained(
        frame: &Self::Frame,
        source: &Self::Buffer,
        source_at: u64,
        into: &Self::Buffer,
        into_at: u64,
        len: u64,
    ) -> Result<()>;
}

/// The device a load binds, as the shared engine reaches it.
pub trait Context<B: Backend> {
    /// A frame to record into.
    fn frame(&self) -> Result<B::Frame>;

    /// The bytes one buffer may span.
    fn max_buffer(&self) -> u64;

    fn name(&self) -> &str;

    fn cores(&self) -> u32;

    /// The bytes the device lends this load.
    fn working_set(&self) -> u64;

    /// The bytes of the working set no reservation holds.
    fn left(&self) -> u64;

    /// Makes the device current on this thread.
    fn bind_thread(&self) -> Result<()>;
}

/// Device work recorded on the host, submitted at commit.
pub trait Frame<B: Backend>: 'static {
    fn copy(
        &mut self,
        source: &B::Buffer,
        source_at: u64,
        into: &B::Buffer,
        into_at: u64,
        len: u64,
    ) -> Result<()>;

    /// Copies each `(source_at, into_at, len)` region of `source` into
    /// `into`.
    fn copy_regions(
        &mut self,
        source: &B::Buffer,
        into: &B::Buffer,
        regions: &[(u64, u64, u64)],
    ) -> Result<()> {
        for &(source_at, into_at, len) in regions {
            self.copy(source, source_at, into, into_at, len)?;
        }
        Ok(())
    }

    fn fill_zero(&mut self, into: &B::Buffer, at: u64, len: u64) -> Result<()>;

    /// Writes `bytes` at `at` of `into` in the frame's order: no work in
    /// flight reads it changed, and no work of this frame reads it
    /// unwritten.
    fn update(&self, into: &B::Buffer, at: u64, bytes: &[u8]) -> Result<()> {
        self.flush_settled()?;
        into.write_shared(at, bytes)
    }

    /// Submits what the frame holds so far; it records on.
    fn flush(&self) -> Result<()>;

    /// [`Frame::flush`], then every frame submitted before has landed too:
    /// the host may rewrite memory any of them reads.
    fn flush_settled(&self) -> Result<()>;

    /// Submits and waits.
    fn commit(self) -> Result<()>;

    /// Submits; `on_done` runs once the frame lands.
    fn commit_async(self, on_done: Option<OnDone>) -> Result<B::Pending>;
}

/// How a backend runs kernels-slang's entries on its device: the tactics its
/// tuning profile and device facts pick. `tier` is the highest tier the
/// adapter runs, where a backend reads it.
pub trait Tactics: std::fmt::Debug {
    /// The profile the device resolved to.
    fn profile(&self) -> &profile::Profile;

    /// The `kernel.qmm` ladder at this shape.
    fn qmm_ladder(
        &self,
        rows: u32,
        columns: u32,
        contraction: u32,
        bank: kernels_slang::Bank,
    ) -> Vec<kernels_slang::linear::qmm::Qmm>;

    fn kquant(&self) -> kernels_slang::linear::kquant::Tactic;

    fn gemm(&self, tier: kernels_slang::Tier) -> kernels_slang::linear::gemm::Tactic;

    /// The tiles and sort a batched route of `pairs` over `experts` runs at.
    fn batch(&self, pairs: u32, experts: u32) -> kernels_slang::linear::moe::select::Batch;

    /// The tier a grouped or routed matmul runs at.
    fn routed_tier(&self, tier: kernels_slang::Tier) -> kernels_slang::Tier;

    /// The tier a LoRA correction runs at.
    fn lora_tier(&self) -> kernels_slang::Tier;

    /// Paged attention for `head_dim`-wide heads.
    fn sdpa(&self, tier: kernels_slang::Tier, head_dim: u32) -> kernels_slang::attn::sdpa::Tactic;

    /// Whether the device binds a split decode's planes (and its LSE's).
    fn binds_split(&self, _lse: bool) -> bool {
        true
    }

    /// The recurrences' fold and register-state threshold.
    fn recurrent(&self) -> kernels_slang::ssm::gdn_recurrent::Tactic;

    fn moe_tile_rows(&self, pairs: u32, experts: u32) -> u32 {
        if experts == 0 {
            return 1;
        }
        let at = crate::tuning::MoeTileRowsAt {
            per_expert: i64::from(pairs / experts),
        };
        self.profile().pick::<crate::tuning::MoeTileRows>(&at).0
    }

    /// The rows a batched route sorts into: every touched expert's run
    /// padded to whole `kernel.moe.tile_rows` tiles; the pairs themselves
    /// where the route stays unsorted.
    fn moe_sorted_rows(&self, pairs: u32, experts: u32) -> u32 {
        let tile = self.moe_tile_rows(pairs, experts);
        if tile <= 1 {
            return pairs;
        }
        let touched = pairs.min(experts);
        let bound = pairs.saturating_add(touched.saturating_mul(tile - 1));
        bound.div_ceil(tile) * tile
    }

    fn sdpa_tile_rows_per_request(&self) -> u32 {
        self.profile()
            .get::<crate::tuning::SdpaTileRowsPerRequest>()
            .0
    }

    fn sdpa_split_groups(&self) -> u32;

    fn sdpa_split_min_keys(&self) -> u32 {
        self.profile().get::<crate::tuning::DecodeSplitMinKeys>().0
    }
}

/// A submitted frame.
pub trait Pending: std::fmt::Debug {
    /// Blocks until the frame lands.
    fn wait(&self) -> Result<()>;
}

/// A device reservation as the shared engine moves bytes through it.
pub trait Buffer<B: Backend>: Clone + std::fmt::Debug + Send + Sync {
    fn with(device: &B::Context, bytes: u64, kind: Memory) -> Result<Self>;

    fn zeroed(device: &B::Context, bytes: u64) -> Result<Self> {
        Self::with(device, bytes, Memory::Device)
    }

    fn host(device: &B::Context, bytes: u64) -> Result<Self> {
        Self::with(device, bytes, Memory::Host)
    }

    /// The bytes asked for.
    fn bytes(&self) -> u64;

    /// The bytes the reservation spans (at least [`Buffer::bytes`]).
    fn slab_size(&self) -> u64;

    /// A name of the reservation, equal across clones.
    fn slab_id(&self) -> u64;

    /// Whether the host maps the reservation.
    fn is_host(&self) -> bool;

    /// The address the reservation is mapped at, where it is.
    fn mapped_ptr(&self) -> Option<usize>;

    fn span(&self, offset: u64, len: u64) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= self.bytes() => Ok(()),
            _ => Err(Fault::Ceiling {
                what: "bytes of a device reservation",
                need: offset.saturating_add(len),
                have: self.bytes(),
            }),
        }
    }

    fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()>;

    /// [`Buffer::write`] through a shared reference.
    fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<()>;

    fn zero_span(&mut self, offset: u64, len: u64) -> Result<()>;

    fn read(&self, offset: u64, into: &mut [u8]) -> Result<()>;

    /// Reads `(into, from, len)` jobs of `file` into the reservation.
    fn write_from_file(
        &mut self,
        file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()>;
}
