//! A backend that reserves nothing and fires nothing: the load's own code
//! run against it leaves the books a real load would leave and the shell a
//! real load would raise, and a fire walked through that shell names its
//! kernels ([`crate::naming`]) where a device would dispatch them. A load
//! runs it first, so what it names is held to the adapter's limits and
//! built before the device holds a byte of the model.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::{Backend, Buffer, Context, Frame, Memory, OnDone, Pending};
use crate::books::{Books, Owner};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::naming::Names;
use crate::serve::{Launch, Plan, Prepared, Serve, Shell};

/// `B` with nothing behind it: its name, its tuning, its fusions and its
/// kernels' target, and no device.
pub struct Named<B>(PhantomData<fn() -> B>);

impl<B> Clone for Named<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for Named<B> {}

impl<B> Default for Named<B> {
    fn default() -> Self {
        Named(PhantomData)
    }
}

impl<B> std::fmt::Debug for Named<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Named")
    }
}

impl<B> PartialEq for Named<B> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<B> Eq for Named<B> {}

impl<B> std::hash::Hash for Named<B> {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {}
}

/// What a load reads of its device, as the real one answered when the
/// naming load began, and books of its own that move no process tally.
#[derive(Debug)]
pub struct Facts {
    name: String,
    max_buffer: u64,
    cores: u32,
    capacity: u64,
    outside: u64,
    memory: engine::DeviceMemory,
    table_align: u64,
    tier: kernels_slang::Tier,
    books: Arc<Books>,
}

impl Facts {
    /// `device`'s facts, its books' device bytes carried as bytes outside
    /// these: what the device already holds is not this load's to place.
    pub fn of<B: Serve>(device: &B::Context) -> Facts {
        Facts {
            name: device.name().to_string(),
            max_buffer: device.max_buffer(),
            cores: device.cores(),
            capacity: device.capacity(),
            outside: device.outside().saturating_add(device.books().device()),
            memory: device.memory(),
            table_align: B::table_align(device),
            tier: B::tier(device),
            books: Arc::new(Books::apart()),
        }
    }
}

impl<B: Serve> Context<Named<B>> for Facts {
    fn frame(&self) -> Result<NoFrame> {
        Ok(NoFrame)
    }

    fn spanned(&self) -> Result<NoFrame> {
        Ok(NoFrame)
    }

    /// Nothing is ever submitted.
    fn idle(&self) -> bool {
        true
    }

    fn max_buffer(&self) -> u64 {
        self.max_buffer
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn cores(&self) -> u32 {
        self.cores
    }

    fn capacity(&self) -> u64 {
        self.capacity
    }

    fn outside(&self) -> u64 {
        self.outside
    }

    fn memory(&self) -> engine::DeviceMemory {
        self.memory
    }

    fn books(&self) -> &Books {
        &self.books
    }

    fn bind_thread(&self) -> Result<()> {
        Ok(())
    }
}

static SLABS: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct Slab {
    id: u64,
    bytes: u64,
    /// The books a device-kind reservation stands in, and under whom.
    charged: Option<(Arc<Books>, Owner)>,
}

impl Drop for Slab {
    fn drop(&mut self) {
        if let Some((books, owner)) = &self.charged {
            books.give(*owner, self.bytes);
        }
    }
}

/// A reservation of nothing: its bytes in the books, as a device kind's
/// are on every backend, and zeros to whoever reads it.
#[derive(Clone, Debug)]
pub struct NoBuffer(Arc<Slab>);

impl<B: Serve> Buffer<Named<B>> for NoBuffer {
    fn with(device: &Facts, bytes: u64, kind: Memory, owner: Owner) -> Result<Self> {
        if bytes > device.max_buffer {
            return Err(Fault::Ceiling {
                what: "bytes of one device reservation",
                need: bytes,
                have: device.max_buffer,
            });
        }
        let charged = if kind == Memory::Device {
            let left = Context::<Named<B>>::left(device);
            if bytes > left {
                return Err(Fault::Ceiling {
                    what: "device memory",
                    need: bytes,
                    have: left,
                });
            }
            device.books.take(owner, bytes);
            Some((Arc::clone(&device.books), owner))
        } else {
            None
        };
        Ok(NoBuffer(Arc::new(Slab {
            id: SLABS.fetch_add(1, Ordering::Relaxed),
            bytes,
            charged,
        })))
    }

    fn bytes(&self) -> u64 {
        self.0.bytes
    }

    fn slab_size(&self) -> u64 {
        self.0.bytes
    }

    fn slab_id(&self) -> u64 {
        self.0.id
    }

    fn is_host(&self) -> bool {
        false
    }

    fn mapped_ptr(&self) -> Option<usize> {
        None
    }

    fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        Buffer::<Named<B>>::span(self, offset, bytes.len() as u64)
    }

    fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        Buffer::<Named<B>>::span(self, offset, bytes.len() as u64)
    }

    fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        Buffer::<Named<B>>::span(self, offset, len)
    }

    fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        Buffer::<Named<B>>::span(self, offset, into.len() as u64)?;
        into.fill(0);
        Ok(())
    }

    fn write_from_file(
        &mut self,
        _file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        _threads: usize,
    ) -> Result<()> {
        jobs.iter()
            .try_for_each(|&(into, _, len)| Buffer::<Named<B>>::span(self, into, len))
    }
}

/// A frame that records nothing.
#[derive(Debug)]
pub struct NoFrame;

impl<B: Serve> Frame<Named<B>> for NoFrame {
    fn copy(
        &mut self,
        source: &NoBuffer,
        source_at: u64,
        into: &NoBuffer,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        Buffer::<Named<B>>::span(source, source_at, len)?;
        Buffer::<Named<B>>::span(into, into_at, len)
    }

    fn fill_zero(&mut self, into: &NoBuffer, at: u64, len: u64) -> Result<()> {
        Buffer::<Named<B>>::span(into, at, len)
    }

    fn dispatches(&self) -> u64 {
        0
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }

    fn flush_settled(&self) -> Result<()> {
        Ok(())
    }

    fn commit(self) -> Result<()> {
        Ok(())
    }

    fn commit_async(self, on_done: Option<OnDone>) -> Result<Landed> {
        if let Some(done) = on_done {
            done(None);
        }
        Ok(Landed)
    }

    /// A frame of nothing runs on no device: no time to read.
    fn commit_spanned(self) -> Result<Option<u64>> {
        Ok(None)
    }
}

/// A frame of nothing has landed.
#[derive(Debug)]
pub struct Landed;

impl Pending for Landed {
    fn wait(&self) -> Result<()> {
        Ok(())
    }

    fn landed(&self) -> bool {
        true
    }
}

impl<B: Serve> Backend for Named<B> {
    const NAME: &'static str = B::NAME;

    /// No host mapping stands behind a reservation of nothing: host chunks
    /// are process memory, as on a backend that maps none.
    const MAPS_HOST: bool = false;

    const COPY_GRAIN: u64 = B::COPY_GRAIN;

    const STORAGE: checkpoint::types::BackendKind = B::STORAGE;

    /// A naming load reserves nothing to land bytes in: its weights are
    /// where the plan lays them, and the walk reads none.
    const LANDS: bool = false;

    type Context = Facts;

    type Buffer = NoBuffer;

    type Frame = NoFrame;

    type Pending = Landed;

    type Tuning = B::Tuning;

    fn is_device(domain: engine::transfer::MemoryDomain) -> bool {
        B::is_device(domain)
    }

    fn stage_rows(
        staging: &NoBuffer,
        at: u64,
        len: u64,
        fill: impl FnOnce(usize) -> Result<()>,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        Buffer::<Named<B>>::span(staging, at, len)?;
        // The rows are staged into memory that lives for the fill.
        let mut rows = vec![0u8; len as usize];
        fill(rows.as_mut_ptr() as usize)
    }

    fn copy_unretained(
        _frame: &NoFrame,
        source: &NoBuffer,
        source_at: u64,
        into: &NoBuffer,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        Buffer::<Named<B>>::span(source, source_at, len)?;
        Buffer::<Named<B>>::span(into, into_at, len)
    }
}

/// The pipelines of a backend that builds none: the kernels its walks ask
/// for.
pub struct Asked<B>(pub Names, PhantomData<fn() -> B>);

impl<B: Serve> Default for Asked<B> {
    fn default() -> Self {
        Asked(Names::new(B::TARGET), PhantomData)
    }
}

/// The naming sink over a walk. As a backend's own sink does, it is this
/// thread's recording frame for as long as it lives: what a dispatch does
/// in the frame's order beside its kernel (a staged table, a settled read)
/// reaches the frame that records nothing.
pub struct Naming<'f>(&'f Names, crate::probe::Recording);

impl Drop for Naming<'_> {
    fn drop(&mut self) {
        crate::probe::restore(self.1);
    }
}

impl kernels_slang::Encode for Naming<'_> {
    fn fire(
        &self,
        fire: kernels_slang::Fire,
        args: &[kernels_slang::ArgValue],
    ) -> std::result::Result<(), kernels_slang::Error> {
        self.0.fire(fire, args)
    }

    fn absent(&self) -> std::result::Result<kernels_slang::ArgValue, kernels_slang::Error> {
        self.0.absent()
    }

    fn target(&self) -> kernels_slang::Target {
        self.0.target()
    }

    fn naming(&self) -> Option<&dyn kernels_slang::Naming> {
        self.0.naming()
    }
}

impl<B: Serve> Serve for Named<B> {
    type DeviceBoot = B::DeviceBoot;

    type Handed = B::Handed;

    type Pipelines = Asked<B>;

    type Sink<'f> = Naming<'f>;

    type Guests = std::convert::Infallible;

    type Shelf = ();

    type Flight = ();

    type Keep = std::convert::Infallible;

    type Probed = ();

    type Stamps = ();

    const FUSIONS: &'static [poem_compiler::fuse::Rule] = B::FUSIONS;

    const MAX_BUFFER_SOURCE: &'static str = B::MAX_BUFFER_SOURCE;

    const RECORDS: bool = B::RECORDS;

    const CHANNEL_CELLS: bool = B::CHANNEL_CELLS;

    const READS_SPAN: bool = B::READS_SPAN;

    const NAMES: bool = true;

    const TARGET: kernels_slang::Target = B::TARGET;

    fn table_align(device: &Facts) -> u64 {
        device.table_align
    }

    /// A naming load is raised over the facts of a device already open.
    fn open(
        _boot: &B::DeviceBoot,
        _allowance: engine::Allowance,
        _handed: Option<B::Handed>,
    ) -> Result<(Facts, B::Tuning)> {
        Err(Fault::Deviceless)
    }

    fn time_kernels(
        _device: &Self::Context,
        _tuning: &Self::Tuning,
        _tier: kernels_slang::Tier,
        _max_submit: Option<std::time::Duration>,
        _launch: f64,
        kernels: &[kernels_slang::linear::gemm::Tactic],
        _pieces: &[u32],
        _plane: (u32, u32),
        _rows: u32,
    ) -> Result<Vec<Vec<f64>>> {
        Ok(vec![Vec::new(); kernels.len()])
    }

    fn save_rates(
        _device: &Self::Context,
        _boot: &Self::DeviceBoot,
        _rates: &profile::fire::Rates,
    ) -> Result<()> {
        Ok(())
    }

    fn time_for_bound(
        _device: &Facts,
        _tuning: &B::Tuning,
        _tier: kernels_slang::Tier,
        _max_submit: std::time::Duration,
        _at: &crate::rates::ForBound,
    ) -> Result<crate::rates::BoundTimes> {
        Err(crate::error::Fault::Unbound {
            what: "a submit bound's timing on a naming load, which runs nothing on a device"
                .into(),
        })
    }

    fn rate_shapes(
        _device: &Facts,
        _boot: &B::DeviceBoot,
        tuning: B::Tuning,
        _shapes: &[(u32, u32)],
        _model_bytes: u64,
        _top_rows: u32,
    ) -> Result<B::Tuning> {
        Ok(tuning)
    }

    fn chunk_bytes(boot: &B::DeviceBoot) -> Option<u64> {
        B::chunk_bytes(boot)
    }

    fn committed_covers() -> &'static [&'static str] {
        B::committed_covers()
    }

    fn measuring(boot: &B::DeviceBoot) -> engine::rates::Measuring {
        B::measuring(boot)
    }

    fn submit_bound(boot: &Self::DeviceBoot) -> Option<std::time::Duration> {
        B::submit_bound(boot)
    }

    fn tier(device: &Facts) -> kernels_slang::Tier {
        device.tier
    }

    fn compiled(_pipelines: &Asked<B>) -> u64 {
        0
    }

    fn built_since(_pipelines: &Asked<B>, _from: u64) -> Vec<&'static str> {
        Vec::new()
    }

    fn hold(_device: &Facts, _unit: &'static str, _key: &'static str) -> Result<()> {
        Ok(())
    }

    fn build(
        _device: &Facts,
        _pipelines: &Asked<B>,
        _unit: &'static str,
        _key: &'static str,
    ) -> Result<()> {
        Ok(())
    }

    fn sink<'f>(
        _device: &'f Facts,
        frame: &'f NoFrame,
        pipelines: &'f Asked<B>,
        _handles: &'f Handles<Named<B>>,
    ) -> Naming<'f> {
        Naming(
            &pipelines.0,
            crate::probe::set_frame(std::ptr::from_ref(frame)),
        )
    }

    const BINDS_AT_LOAD: bool = false;

    fn bind_frame(_device: &Facts) -> Result<NoFrame> {
        Err(Fault::Unbound {
            what: "a bind walk on the naming load, which keeps nothing a dispatch binds".into(),
        })
    }

    fn bound(_pipelines: &Asked<B>) -> u64 {
        0
    }

    fn walk_frame(_device: &Facts, _keep: bool, _timed: Option<&()>) -> Result<NoFrame> {
        Ok(NoFrame)
    }

    /// A naming load runs nothing on a device: no timestamps to lay.
    fn lay_stamps(_device: &Facts, _pieces: u32) -> Result<()> {
        Ok(())
    }

    /// A naming load runs nothing on a device: no fire to time.
    fn fire_timing(_device: &Facts) -> Result<crate::submit::FireTiming> {
        Err(crate::error::Fault::Unbound {
            what: "a fire's device time on a naming load, which runs nothing on a device".into(),
        })
    }

    /// A naming load runs nothing on a device: no probe to lay.
    fn record_probe(_device: &Facts, _record: &dyn Fn(&NoFrame) -> Result<()>) -> Result<()> {
        Ok(())
    }

    fn replay_probe(
        _device: &Facts,
        _probed: &(),
        _record: &dyn Fn(&NoFrame) -> Result<()>,
    ) -> Result<Option<u64>> {
        Ok(None)
    }

    fn shelf(
        _device: &Facts,
        _pipelines: &Self::Pipelines,
        _readout: &[NoBuffer],
        _weights_stream: bool,
        _guest: crate::serve::GuestRoom,
    ) -> Result<()> {
        Ok(())
    }

    fn forget(_shelf: &mut ()) {}

    fn grown(_shell: &mut Shell<Named<B>>, _arm: usize, _bytes: u64) -> Result<()> {
        Ok(())
    }

    fn plan(
        _shell: &mut Shell<Named<B>>,
        _prepared: &mut Prepared<'_, Named<B>>,
    ) -> Result<Plan<Named<B>>> {
        Ok(Plan::Walk {
            keep: None,
            split: false,
        })
    }

    fn submit(
        _shell: &mut Shell<Named<B>>,
        frame: NoFrame,
        _prepared: &mut Prepared<'_, Named<B>>,
        _launch: Launch<'_, Named<B>>,
    ) -> Result<Landed> {
        Frame::<Named<B>>::commit_async(frame, None)
    }

    fn fly(_pending: Landed, _fed: Option<Landed>) {}

    fn reap(_shell: &mut Shell<Named<B>>) -> Result<()> {
        Ok(())
    }

    fn harvest_one(_shell: &mut Shell<Named<B>>) -> Result<()> {
        Ok(())
    }
}
