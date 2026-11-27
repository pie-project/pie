//! The API half a backend crate supplies: its device, its buffers, its
//! frames. Everything else in this crate is generic over [`Backend`] and is
//! monomorphized per backend, so a call through these traits is static.

use crate::books::{Books, Owner};
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

    /// Whether the state pool's segments commit and release while serving.
    /// One that does not lays its whole pool at load and states it
    /// `Growth::Fixed`: no `yield` applies to it, and the runtime never
    /// moves it.
    const RESIZES_POOL: bool = true;

    /// Whether a load lands the checkpoint's bytes in what it reserves.
    /// One that reserves nothing takes where each plane lies and which the
    /// checkpoint publishes from the load plan, and reads no byte.
    const LANDS: bool = true;

    /// The device one load binds.
    type Context: Context<Self>;

    /// One device reservation; clones share it.
    type Buffer: Buffer<Self>;

    /// Device work recorded and not yet submitted.
    type Frame: Frame<Self>;

    /// A submitted frame, until it lands.
    type Pending: Pending;

    /// The device's tuning profile, as the dispatch reads it.
    type Tuning: Tactics + Clone;

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

    /// Submits what this thread's recording frame holds as a split step's
    /// segment, unawaited ([`Frame::submit_segment`]), if one is recording.
    fn submit_recording() -> Result<()> {
        Self::with_frame(Frame::submit_segment).unwrap_or(Ok(()))
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

    /// A frame whose device time [`Frame::commit_spanned`] reads back: a
    /// timestamp written before its work and one after it, where the
    /// device's queue writes timestamps; a plain frame where it writes none.
    fn spanned(&self) -> Result<B::Frame>;

    /// Whether every submission made on the device so far has landed, read
    /// now without waiting: the device's queue is empty.
    fn idle(&self) -> bool;

    /// The bytes one buffer may span.
    fn max_buffer(&self) -> u64;

    /// The segments the state pool is laid in, as the kernels this device
    /// runs bind it: [`kernels_slang::SEGMENTS`], or one where a stage
    /// binds too few storage buffers for four beside a kernel's own (the
    /// kernels' one-segment form, `kernels/prim/pool.slang`).
    fn pool_segments(&self) -> u32 {
        kernels_slang::SEGMENTS as u32
    }

    /// Whether two arguments of one dispatch may bind overlapping views of
    /// a buffer ([`kernels_slang::Encode::binds_overlap`]): false where the
    /// device's API refuses them (a browser's WebGPU).
    fn binds_overlap(&self) -> bool {
        true
    }

    fn name(&self) -> &str;

    fn cores(&self) -> u32;

    /// Whether the state pool may commit and release segments while serving
    /// on this device, where its backend's pool resizes
    /// ([`Backend::RESIZES_POOL`]): a device that cannot rebind what a
    /// recording binds of the pool lays it whole, as it states at load.
    fn resizes_pool(&self) -> bool {
        true
    }

    /// The workgroups the device holds at once, as the load laid them
    /// (measured, stored, committed or estimated): what a guest step's grid
    /// strides over. `None` before the load has.
    fn resident(&self) -> Option<u32> {
        None
    }

    /// What the device has for this load, measured now: what it holds for
    /// the load (every reservation and what the driver holds beside them)
    /// and what it has free, capped by the operator's `budget_device`.
    fn capacity(&self) -> u64;

    /// What the driver holds for this process beside every reservation, as
    /// measured now; zero where the device does not say.
    fn outside(&self) -> u64;

    /// The reader of what the host can still give this process, where the
    /// device shares the host's memory and no budget of the device's says
    /// what it has ([`engine::Engine::host_room`]). `None`, the default:
    /// the device has memory of its own.
    fn host_room(&self) -> Option<engine::HostRoom> {
        None
    }

    /// Whose memory the device's is: where it is the host's, the load
    /// states no host tier ([`engine::DeviceMemory::host_tier`]).
    fn memory(&self) -> engine::DeviceMemory;

    /// The bytes of the capacity neither a reservation nor the driver holds.
    fn left(&self) -> u64 {
        self.capacity()
            .saturating_sub(self.books().device())
            .saturating_sub(self.outside())
    }

    /// The device's books: every reservation under its owner.
    fn books(&self) -> &Books;

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

    /// [`Frame::copy_regions`] for the rows a fire reads out, which move
    /// by the fire's lanes: at most `most` regions for any fire of the
    /// frame's shape. A backend whose recordings keep what a fire changes
    /// apart from their commands moves them so.
    fn copy_rows(
        &mut self,
        source: &B::Buffer,
        into: &B::Buffer,
        regions: &[(u64, u64, u64)],
        most: usize,
    ) -> Result<()> {
        let _ = most;
        self.copy_regions(source, into, regions)
    }

    fn fill_zero(&mut self, into: &B::Buffer, at: u64, len: u64) -> Result<()>;

    /// Writes `bytes` at `at` of `into` in the frame's order: no work in
    /// flight reads it changed, and no work of this frame reads it
    /// unwritten.
    fn update(&self, into: &B::Buffer, at: u64, bytes: &[u8]) -> Result<()> {
        self.flush_settled()?;
        into.write_shared(at, bytes)
    }

    /// The dispatches the frame has encoded so far.
    fn dispatches(&self) -> u64;

    /// `(taken, room)`: the bytes the frame's dispatches have taken so far of
    /// the region a frame writes its dispatches' parameters to, and that
    /// region's bytes, where the backend bounds them so; `None` where it
    /// bounds none.
    fn uniform_room(&self) -> Option<(u64, u64)> {
        None
    }

    /// Submits what the frame holds so far; it records on.
    fn flush(&self) -> Result<()>;

    /// Ends what the frame holds so far as a segment of its own submission:
    /// a split step's (a run of layers), which the platform's bound wants
    /// short. The segments go to the device with the frame, each its own
    /// submission, back to back, so the device runs them without waiting
    /// on the host's encoding between them; the work recorded next is
    /// ordered after each. The host writes nothing a segment reads in
    /// between (no staging between a split step's layers rides on it).
    /// Where the backend cannot, [`Frame::flush`].
    fn submit_segment(&self) -> Result<()> {
        self.flush()
    }

    /// [`Frame::flush`], then every frame submitted before has landed too:
    /// the host may rewrite memory any of them reads.
    fn flush_settled(&self) -> Result<()>;

    /// Submits and waits.
    fn commit(self) -> Result<()>;

    /// Submits; `on_done` runs once the frame lands.
    fn commit_async(self, on_done: Option<OnDone>) -> Result<B::Pending>;

    /// Submits and waits: the device's own time across the frame's work,
    /// ns, from the timestamps a [`Context::spanned`] frame wrote; `None`
    /// for any other frame, or where the device writes no timestamps.
    fn commit_spanned(self) -> Result<Option<u64>>;
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

    /// Every `kernel.qmm` rung some batch of 1 to `capacity` rows reaches
    /// at this shape: [`Tactics::qmm_ladder`]'s answers over those rows.
    fn qmm_rungs(
        &self,
        capacity: u32,
        columns: u32,
        contraction: u32,
        bank: kernels_slang::Bank,
    ) -> Vec<kernels_slang::linear::qmm::Qmm>;

    fn kquant(&self) -> kernels_slang::linear::kquant::Tactic;

    /// The tiled dense matmul tactic at `rows` activation rows; whether a
    /// rung takes the vector kernels instead is the load's choice
    /// ([`crate::dense`]).
    fn gemm(&self, tier: kernels_slang::Tier, rows: u32) -> kernels_slang::linear::gemm::Tactic;

    /// Every dense matmul tactic some fire of 1 to `capacity` rows takes:
    /// [`Tactics::gemm`]'s answers over those rows, each once.
    fn gemm_tactics(
        &self,
        tier: kernels_slang::Tier,
        capacity: u32,
    ) -> Vec<kernels_slang::linear::gemm::Tactic>;

    /// The tiles and sort a batched route of `pairs` over `experts` runs at.
    fn batch(&self, pairs: u32, experts: u32) -> kernels_slang::linear::moe::select::Batch;

    /// The tier a grouped or routed matmul runs at.
    fn routed_tier(&self, tier: kernels_slang::Tier) -> kernels_slang::Tier;

    /// The tier a LoRA correction runs at.
    fn lora_tier(&self) -> kernels_slang::Tier;

    /// Paged attention for `head_dim`-wide heads.
    fn sdpa(&self, tier: kernels_slang::Tier, head_dim: u32) -> kernels_slang::attn::sdpa::Tactic;

    /// Whether the device binds the split walk of a decode of
    /// `head_dim`-wide heads (`lse` where it lands its LSE), its planes
    /// beside the whole walk's.
    fn binds_split(&self, _head_dim: u32, _lse: bool) -> bool {
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

    /// Every row tile a routed matmul of up to `pairs` pairs sorts into:
    /// what [`Tactics::moe_tile_rows`] answers some count of them.
    fn moe_tiles(&self, pairs: u32, experts: u32) -> Vec<u32> {
        if experts == 0 {
            return vec![1];
        }
        let cell = [profile::Set::span(0, i64::from(pairs / experts))];
        let mut tiles: Vec<u32> = Vec::new();
        for tile in self
            .profile()
            .ladder::<crate::tuning::MoeTileRows>()
            .meeting(&cell)
        {
            if !tiles.contains(&tile.0) {
                tiles.push(tile.0);
            }
        }
        tiles
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

    /// The workgroup memory a kernel may take, as the device states it
    /// (`maxComputeSharedMemorySize`, `maxComputeWorkgroupStorageSize`):
    /// what a kernel staged in it sizes its form by.
    fn workgroup_bytes(&self) -> u32;

    fn sdpa_split_min_keys(&self) -> u32 {
        self.profile().get::<crate::tuning::DecodeSplitMinKeys>().0
    }
}

/// A submitted frame.
pub trait Pending: std::fmt::Debug {
    /// Blocks until the frame lands.
    fn wait(&self) -> Result<()>;

    /// Whether the frame has landed, read now without waiting.
    fn landed(&self) -> bool;
}

/// A device reservation as the shared engine moves bytes through it.
pub trait Buffer<B: Backend>: Clone + std::fmt::Debug + Send + Sync {
    /// A reservation of `bytes` of `kind`, entered in the device's books
    /// under `owner` while it stands.
    fn with(device: &B::Context, bytes: u64, kind: Memory, owner: Owner) -> Result<Self>;

    fn zeroed(device: &B::Context, bytes: u64, owner: Owner) -> Result<Self> {
        Self::with(device, bytes, Memory::Device, owner)
    }

    fn host(device: &B::Context, bytes: u64, owner: Owner) -> Result<Self> {
        Self::with(device, bytes, Memory::Host, owner)
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
