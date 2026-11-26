//! Device memory kept behind a fence on the compute streams and handed out
//! again once it lands, rather than given back to the driver and asked for
//! anew.
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicPtr, Ordering};

use crate::device::rt;

/// `cudaEventDisableTiming`, as HIP's is too.
pub(crate) const DISABLE_TIMING: u32 = 2;

const ORDINALS: usize = 64;

/// Each ordinal's latest adopted compute stream: whether it has one live.
static STREAMS: [AtomicPtr<core::ffi::c_void>; ORDINALS] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; ORDINALS];

/// A live context's compute stream, and while a capture is open on it, the
/// mark of what it held before the capture began.
struct Live {
    ordinal: i32,
    stream: usize,
    captured: Option<std::sync::Arc<Mark>>,
}

/// The live contexts' compute streams, in adoption order: a stream is
/// recorded on only while this is read-held, and retired (and destroyed),
/// or begins or ends a capture, only while it is write-held, so no event is
/// recorded on a destroyed stream, or on one being captured, whichever
/// context of the process goes first.
static LIVE: std::sync::RwLock<Vec<Live>> = std::sync::RwLock::new(Vec::new());

/// A context's compute stream, every fire of `ordinal` it runs launching on
/// it.
pub fn adopt_stream(ordinal: i32, stream: *mut core::ffi::c_void) {
    let mut live = LIVE.write().expect("live lock poisoned");
    live.push(Live {
        ordinal,
        stream: stream as usize,
        captured: None,
    });
    if let Some(slot) = usize::try_from(ordinal).ok().and_then(|at| STREAMS.get(at)) {
        slot.store(stream, Ordering::Release);
    }
}

/// Takes `stream` out of `ordinal`'s adoption (another live context's
/// stream, if any, takes its place) and runs `destroy` before any recorder
/// can see the stream again.
pub fn retire_stream(ordinal: i32, stream: *mut core::ffi::c_void, destroy: impl FnOnce()) {
    let mut live = LIVE.write().expect("live lock poisoned");
    if let Some(at) = live
        .iter()
        .position(|entry| (entry.ordinal, entry.stream) == (ordinal, stream as usize))
    {
        live.remove(at);
    }
    let next = live
        .iter()
        .rev()
        .find(|entry| entry.ordinal == ordinal)
        .map_or(core::ptr::null_mut(), |entry| {
            entry.stream as *mut core::ffi::c_void
        });
    if let Some(slot) = usize::try_from(ordinal).ok().and_then(|at| STREAMS.get(at)) {
        slot.store(next, Ordering::Release);
    }
    // What the stream enqueued lands before it goes, and no kept fence
    // outlives it holding a mark of it: a mark's event belongs to the stream
    // it was recorded on, and a query of it once the stream is destroyed
    // reads freed state (on ROCm, a lock no thread holds). Every fence the
    // stores can see then marks only live streams.
    // SAFETY: the stream is live till `destroy`, and no recorder can reach
    // it while `LIVE` is write-held.
    unsafe { rt::cudaStreamSynchronize(stream.cast()) };
    forget_stream(stream);
    destroy();
}

/// Every mark a store or a staging ring keeps of `stream`, forgotten: the
/// fences kept memory waits on, and on ROCm the staging rings' slots (the
/// pageable ring's and the attention plans'), whose events belong to the
/// stream they were recorded on. The stream must still be live and drained.
fn forget_stream(stream: *mut core::ffi::c_void) {
    let at = stream as usize;
    pool::forget_stream(at);
    heap::forget_stream(at);
    #[cfg(feature = "rocm")]
    {
        crate::device::hip::held::forget_stream(at);
        kernels_cuda::attn::plan::forget_stream(stream);
    }
}

/// Destroys a stream no context adopted (a side, notify, tap, swap, fan or
/// staging lane stream): drained, every mark of it forgotten
/// ([`forget_stream`]), then destroyed, so no later query reaches an event
/// it recorded.
///
/// # Safety
///
/// `stream` is a live stream nothing launches on again.
pub unsafe fn destroy_unadopted_stream(stream: *mut core::ffi::c_void) {
    if stream.is_null() {
        return;
    }
    // SAFETY: as the caller promises.
    unsafe { rt::cudaStreamSynchronize(stream.cast()) };
    forget_stream(stream);
    // SAFETY: as the caller promises; nothing below reaches it again.
    let _ = unsafe { rt::cudaStreamDestroy(stream.cast()) };
}

/// Opens a capture on `stream` with `begin`, the work it holds before the
/// capture marked first. Work captured does not run, so what a fence must
/// wait for on a capturing stream is what came before the capture; and an
/// event another thread recorded on it would join the capture, and its
/// query (whether the fence landed) invalidate it, so a fence takes this
/// mark instead (`cudaErrorStreamCaptureInvalidated`, 901, on a model's
/// arming while another model on the device freed or allocated).
pub(crate) fn begin_capture(
    stream: *mut core::ffi::c_void,
    begin: impl FnOnce() -> crate::error::Result<()>,
) -> crate::error::Result<()> {
    let mut live = LIVE.write().expect("live lock poisoned");
    let Some(entry) = live
        .iter_mut()
        .find(|entry| entry.stream == stream as usize)
    else {
        return begin();
    };
    let mark = Mark::record(stream).ok_or(crate::error::Fault::Device {
        call: "cudaEventRecord",
        code: 0,
    })?;
    begin()?;
    entry.captured = Some(std::sync::Arc::new(mark));
    Ok(())
}

/// Ends the capture [`begin_capture`] opened on `stream` with `end`; fences
/// record on the stream again from here.
pub(crate) fn end_capture<R>(stream: *mut core::ffi::c_void, end: impl FnOnce() -> R) -> R {
    let mut live = LIVE.write().expect("live lock poisoned");
    let ended = end();
    if let Some(entry) = live
        .iter_mut()
        .find(|entry| entry.stream == stream as usize)
    {
        entry.captured = None;
    }
    ended
}

pub(crate) fn current_device() -> Option<i32> {
    let mut device = 0i32;
    // SAFETY: writes one int.
    let code = unsafe { rt::cudaGetDevice(&raw mut device) };
    (code == rt::cudaError::cudaSuccess).then_some(device)
}

/// Whether `device` has an adopted compute stream.
pub(crate) fn adopted(device: i32) -> bool {
    usize::try_from(device)
        .ok()
        .and_then(|at| STREAMS.get(at))
        .is_some_and(|slot| !slot.load(Ordering::Acquire).is_null())
}

/// An event recorded on a stream, destroyed once its last holder lets go.
struct Mark {
    event: rt::cudaEvent_t,
    /// The stream the event was recorded on, which it belongs to.
    stream: usize,
}

// SAFETY: the event is a process-wide device object, destroyed once.
unsafe impl Send for Mark {}
// SAFETY: as above; a query only reads it.
unsafe impl Sync for Mark {}

impl Mark {
    /// A fresh event recorded on `stream`, which no capture holds.
    fn record(stream: *mut core::ffi::c_void) -> Option<Mark> {
        let mut event: rt::cudaEvent_t = core::ptr::null_mut();
        // SAFETY: creates a fresh event, owned by the mark from here.
        if unsafe { rt::cudaEventCreateWithFlags(&raw mut event, DISABLE_TIMING) }
            != rt::cudaError::cudaSuccess
        {
            return None;
        }
        let mark = Mark {
            event,
            stream: stream as usize,
        };
        // SAFETY: the event is the mark's own; the caller keeps `stream` live.
        (unsafe { rt::cudaEventRecord(event, stream.cast()) } == rt::cudaError::cudaSuccess)
            .then_some(mark)
    }

    fn landed(&self) -> bool {
        // SAFETY: the event is this mark's own.
        unsafe { rt::cudaEventQuery(self.event) == rt::cudaError::cudaSuccess }
    }
}

impl Drop for Mark {
    fn drop(&mut self) {
        // SAFETY: the event is this mark's own, destroyed once.
        unsafe { rt::cudaEventDestroy(self.event) };
    }
}

/// A mark on every live compute stream of a device, taken at once:
/// whatever any of them had enqueued before is done once all have landed.
/// One context's stream alone would not do while another context on the
/// device may still read what is kept. A stream being captured gives the
/// mark of what it held before its capture began ([`begin_capture`]).
pub(crate) struct Fence(Vec<std::sync::Arc<Mark>>);

impl Fence {
    /// `None` where a stream would not record; an empty fence where the
    /// device has no live stream, so nothing of it is airborne.
    pub(crate) fn record(device: i32) -> Option<Fence> {
        let live = LIVE.read().expect("live lock poisoned");
        let mut fence = Fence(Vec::new());
        for entry in live.iter().filter(|entry| entry.ordinal == device) {
            let mark = match &entry.captured {
                Some(before) => std::sync::Arc::clone(before),
                None => std::sync::Arc::new(Mark::record(entry.stream as *mut core::ffi::c_void)?),
            };
            fence.0.push(mark);
        }
        Some(fence)
    }

    pub(crate) fn landed(&self) -> bool {
        self.0.iter().all(|mark| mark.landed())
    }

    /// Lets go of the marks recorded on `stream`, which has landed and is
    /// being retired ([`retire_stream`]).
    fn forget_stream(&mut self, stream: usize) {
        self.0.retain(|mark| mark.stream != stream);
    }
}

/// Endpoint memory kept for the next channel of its shape. Pinning fresh
/// memory takes the driver's own lock, which another process's teardown
/// holds for as long as it unpins (about 0.2 s a GiB of its pinned memory),
/// so on CUDA an instance bound while a peer worker on the host died waited
/// out the whole of it, and every frame behind it; on ROCm freeing pinned or
/// device memory waits for the whole device, and pinning costs about a
/// millisecond a channel, where a burst of requests opens and closes
/// thousands. A returned endpoint is reused only once a fence recorded on
/// the device's compute streams at its return has landed, so no frame that
/// could still read it is airborne; its memory is zeroed again on the way
/// out, as fresh memory would be.
pub mod pool {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{Fence, adopted, current_device};
    use crate::device::rt;
    use crate::device::{Buffer, Pinned};

    pub type Held = (Pinned, Pinned, Option<Buffer>);

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PoolKey {
        device: i32,
        device_only: bool,
        cells: usize,
        /// An instance's commit slot rather than a channel's endpoint.
        commit: bool,
    }

    /// Endpoints of one shape kept at most.
    const PER_KEY: usize = 4096;

    /// Cells past this many bytes are freed rather than kept.
    const LARGEST: usize = 64 << 20;

    const DEVICE_BUDGET: usize = 256 << 20;

    struct Entry {
        held: Held,
        fence: Fence,
    }

    static POOL: Mutex<Option<HashMap<PoolKey, Vec<Entry>>>> = Mutex::new(None);

    static DEVICE_KEPT: AtomicUsize = AtomicUsize::new(0);

    fn device_bytes(held: &Held) -> usize {
        held.2.as_ref().map_or(0, Buffer::bytes)
    }

    /// The key an endpoint of this shape pools under, if any: none on a
    /// device whose compute stream was never adopted, or for huge cells.
    pub fn key(device_only: bool, cells: usize) -> Option<PoolKey> {
        let device = current_device()?;
        if !adopted(device) {
            return None;
        }
        (cells <= LARGEST).then_some(PoolKey {
            device,
            device_only,
            cells,
            commit: false,
        })
    }

    /// The key an instance's `bytes`-wide commit slot pools under.
    pub fn commit_key(bytes: usize) -> Option<PoolKey> {
        let device = current_device()?;
        if !adopted(device) {
            return None;
        }
        Some(PoolKey {
            device,
            device_only: false,
            cells: bytes,
            commit: true,
        })
    }

    /// A kept endpoint of `key`'s shape whose last reader has landed, its
    /// memory zeroed: the device cells on `stream`, the taker's own, ahead
    /// of any fire of it that reads them. With no stream, only an endpoint
    /// with no device cells is handed out.
    pub fn take(key: PoolKey, stream: Option<*mut core::ffi::c_void>) -> Option<Held> {
        if !adopted(key.device) {
            return None;
        }
        let mut pool = POOL.lock().expect("pool lock poisoned");
        let list = pool.as_mut()?.get_mut(&key)?;
        let at = list.iter().position(|entry| {
            entry.fence.landed() && (entry.held.2.is_none() || stream.is_some())
        })?;
        let entry = list.swap_remove(at);
        DEVICE_KEPT.fetch_sub(device_bytes(&entry.held), Ordering::Relaxed);
        drop(pool);
        let (words, mirror, cells) = entry.held;
        words.zero();
        mirror.zero();
        if let (Some(cells), Some(stream)) = (&cells, stream) {
            // SAFETY: the buffer is the entry's, no airborne fire reads it,
            // and the memset is ordered before every later fire of the
            // taker's stream.
            let code = unsafe {
                rt::cudaMemsetAsync(
                    cells.ptr() as *mut core::ffi::c_void,
                    0,
                    cells.bytes(),
                    stream.cast(),
                )
            };
            if code != rt::cudaError::cudaSuccess {
                return None;
            }
        }
        Some((words, mirror, cells))
    }

    /// `count` summed over every kept endpoint's fence.
    pub(super) fn fences(count: &dyn Fn(&Fence) -> usize) -> usize {
        let pool = POOL.lock().expect("pool lock poisoned");
        pool.iter()
            .flat_map(|map| map.values())
            .flatten()
            .map(|entry| count(&entry.fence))
            .sum()
    }

    /// Drops `stream`'s marks from every kept endpoint's fence
    /// ([`super::retire_stream`]).
    pub(super) fn forget_stream(stream: usize) {
        let mut pool = POOL.lock().expect("pool lock poisoned");
        for entry in pool.iter_mut().flat_map(|map| map.values_mut()).flatten() {
            entry.fence.forget_stream(stream);
        }
    }

    /// Keeps `held` for `key`'s shape behind a fence on the compute
    /// streams; frees it (by dropping) when the pool is full or the fence
    /// will not record.
    pub fn give(key: PoolKey, held: Held) {
        let Some(fence) = Fence::record(key.device) else {
            return;
        };
        let mut pool = POOL.lock().expect("pool lock poisoned");
        let list = pool
            .get_or_insert_with(HashMap::new)
            .entry(key)
            .or_default();
        let device = device_bytes(&held);
        if list.len() >= PER_KEY || DEVICE_KEPT.load(Ordering::Relaxed) + device > DEVICE_BUDGET {
            return;
        }
        DEVICE_KEPT.fetch_add(device, Ordering::Relaxed);
        list.push(Entry { held, fence });
    }
}

/// Device memory a dropped [`Buffer`] leaves, kept for the next of its size.
/// Fresh device memory takes the driver's lock, which another process's
/// teardown holds for seconds, so on CUDA a program batch built while a peer
/// worker on the host died stalled the engine thread and every frame behind
/// it; on ROCm `hipFree` waits for the whole device, so a buffer dropped
/// while frames are airborne (a lane table outgrown, a channel closed) would
/// stall it for the frame in flight. Kept behind a fence on the compute
/// streams instead, a buffer is handed out again once that fence lands.
/// Device buffers dropped while their load serves: never freed then (a
/// free waits on the device, and on ROCm on every queue of it), freed once
/// the shell is down ([`release_idle`]). Nothing is kept for reuse.
pub(crate) mod deferred {
    use std::sync::Mutex;

    use crate::device::rt;

    static DROPPED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

    /// `ptr` is freed once the shell is down.
    pub(crate) fn later(ptr: u64) {
        DROPPED.lock().expect("deferred lock poisoned").push(ptr);
    }

    /// Frees every buffer dropped while serving; how many.
    pub(crate) fn free() -> usize {
        let dropped = std::mem::take(&mut *DROPPED.lock().expect("deferred lock poisoned"));
        for &ptr in &dropped {
            // SAFETY: each pointer is a dropped buffer's own, freed once; the
            // shell that served it is down, so no frame reads it.
            let _ = unsafe { rt::cudaFree(ptr as *mut core::ffi::c_void) };
        }
        dropped.len()
    }
}

/// Frees the device buffers dropped while a shell served
/// ([`deferred`]); how many. A load calls it before it measures the card,
/// and a shell once it is down.
pub fn release_idle() -> usize {
    deferred::free()
}

/// A load begins: what the allocator and hipBLASLt took so far is what the
/// device held before it ([`malloced_since`], [`blas_library_since`]).
pub fn heap_mark() {
    #[cfg(feature = "rocm")]
    {
        MALLOC_MARK.store(
            kernels_cuda::hip::rt::malloced_bytes(),
            std::sync::atomic::Ordering::Relaxed,
        );
        LT_MARK.store(
            kernels_cuda::hip::lt::held_bytes(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// The device bytes hipBLASLt took on its own since the last load began
/// (its library's code and each handle's state), where the build has it:
/// what the ledger enters apart from the driver's.
#[must_use]
pub fn blas_library_since() -> u64 {
    #[cfg(feature = "rocm")]
    {
        kernels_cuda::hip::lt::held_bytes()
            .saturating_sub(LT_MARK.load(std::sync::atomic::Ordering::Relaxed))
    }
    #[cfg(not(feature = "rocm"))]
    {
        0
    }
}

#[cfg(feature = "rocm")]
static LT_MARK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The device bytes this process took from the allocator itself since the
/// last load began, where it can tell (`None` where it cannot): what the
/// device holds past them and what it held before is the driver's.
#[must_use]
pub fn malloced_since() -> Option<u64> {
    #[cfg(feature = "rocm")]
    {
        Some(
            kernels_cuda::hip::rt::malloced_bytes()
                .saturating_sub(MALLOC_MARK.load(std::sync::atomic::Ordering::Relaxed)),
        )
    }
    #[cfg(not(feature = "rocm"))]
    {
        None
    }
}

#[cfg(feature = "rocm")]
static MALLOC_MARK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// The marks the kept stores' fences hold of streams no live context owns:
/// none, since a context's close lets go of every mark of its stream
/// ([`retire_stream`]), and a take queries only live streams' events.
#[must_use]
pub fn dead_marks() -> usize {
    let live: Vec<usize> = LIVE
        .read()
        .expect("live lock poisoned")
        .iter()
        .map(|entry| entry.stream)
        .collect();
    let dead = |fence: &Fence| fence.0.iter().filter(|mark| !live.contains(&mark.stream)).count();
    pool::fences(&dead) + heap::fences(&dead)
}

/// Small memory carved from chunks made ahead and kept: pinned (a channel's
/// control words and ring mirror, a commit slot, a tap's rows) and on each
/// device (a program batch's tables, a lane table, staging). Making memory
/// takes the driver's lock, which a dying process's teardown holds for
/// seconds, so a program bound with a shape no earlier one had, or a fire
/// with a new working set, while a peer worker on the host died, stalled
/// every frame behind it. Here the engine's thread carves cells and a thread
/// of the heaps' own makes the chunks they come from, a chunk ahead of need:
/// no lock is held across a driver call, and the engine's thread makes one
/// only when a heap runs dry before its next chunk is made. A cell is a
/// power of two between a kind's smallest and largest; a freed one is handed
/// out again once a fence recorded on the compute streams at its release has
/// landed, so no frame that could still read it is airborne.
pub(crate) mod heap {
    use std::collections::HashMap;
    use std::sync::{Condvar, Mutex, OnceLock};

    use super::{Fence, current_device};
    use crate::device::rt;
    use crate::error::Result;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Kind {
        /// Mapped and portable host memory: one heap serves every device.
        Pinned,
    }

    impl Kind {
        const fn smallest(self) -> usize {
            match self {
                Kind::Pinned => 64,
            }
        }

        const fn largest(self) -> usize {
            match self {
                Kind::Pinned => 1 << 20,
            }
        }

        /// What one chunk holds.
        const fn chunk(self) -> usize {
            match self {
                Kind::Pinned => 64 << 20,
            }
        }

        const fn classes(self) -> usize {
            (self.largest() / self.smallest()).trailing_zeros() as usize + 1
        }

        const fn class_bytes(self, class: usize) -> usize {
            self.smallest() << class
        }
    }

    /// A heap: of a kind, and for device memory, of a device.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct Key {
        kind: Kind,
        /// The device the heap's memory is on; -1 for pinned memory.
        ordinal: i32,
    }

    impl Key {
        fn of(kind: Kind) -> Option<Key> {
            Some(Key {
                kind,
                ordinal: match kind {
                    Kind::Pinned => -1,
                },
            })
        }

        fn make_chunk_for(self, bytes: usize) -> Result<(Cell, usize)> {
            chunk_within(
                self.kind.chunk(),
                bytes,
                |size| self.make_chunk(size),
                || {
                    super::release_idle();
                },
            )
        }

        /// One more chunk of `bytes`; the calling thread is bound to the
        /// heap's device.
        fn make_chunk(self, bytes: usize) -> Result<Cell> {
            match self.kind {
                Kind::Pinned => {
                    let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
                    // SAFETY: `host` is a live local the call fills; the chunk
                    // is kept for the life of the process.
                    unsafe {
                        crate::device::ctx::check(
                            "cudaHostAlloc",
                            rt::cudaHostAlloc(
                                &raw mut host,
                                bytes,
                                rt::cudaHostAllocMapped | rt::cudaHostAllocPortable,
                            ),
                        )?;
                    }
                    let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
                    // SAFETY: `host` is the mapped allocation just made.
                    unsafe {
                        crate::device::ctx::check(
                            "cudaHostGetDevicePointer",
                            rt::cudaHostGetDevicePointer(&raw mut device, host, 0),
                        )?;
                    }
                    Ok(Cell {
                        host: host as usize,
                        device: device as u64,
                        ordinal: -1,
                    })
                }
            }
        }
    }

    fn chunk_within(
        whole: usize,
        bytes: usize,
        mut make: impl FnMut(usize) -> Result<Cell>,
        mut trim: impl FnMut(),
    ) -> Result<(Cell, usize)> {
        let mut size = whole;
        let mut trimmed = false;
        loop {
            match make(size) {
                Ok(cell) => return Ok((cell, size)),
                Err(crate::error::Fault::OutOfMemory { .. }) if !trimmed => {
                    trimmed = true;
                    trim();
                }
                Err(crate::error::Fault::OutOfMemory { .. }) if size / 2 >= bytes => size /= 2,
                Err(fault) => return Err(fault),
            }
        }
    }

    /// A cell: its host address (0 on the device), its device address, and
    /// the device it is on (-1 for pinned memory).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Cell {
        pub host: usize,
        pub device: u64,
        pub ordinal: i32,
    }

    impl Cell {
        const fn at(self, offset: usize) -> Cell {
            Cell {
                host: if self.host == 0 {
                    0
                } else {
                    self.host + offset
                },
                device: self.device + offset as u64,
                ordinal: self.ordinal,
            }
        }
    }

    struct Chunk {
        at: Cell,
        used: usize,
        bytes: usize,
    }

    struct Kept {
        cell: Cell,
        fence: Fence,
    }

    #[derive(Default)]
    struct Heap {
        /// The chunk cells are carved from.
        current: Option<Chunk>,
        /// Chunks made ahead, not yet carved from, and their sizes.
        ready: Vec<(Cell, usize)>,
        /// A chunk is being made for this heap.
        making: bool,
        free: Vec<Vec<Kept>>,
    }

    struct Heaps {
        heaps: Mutex<HashMap<Key, Heap>>,
        /// Heaps their thread should make a chunk for.
        wanted: Mutex<Vec<Key>>,
        wake: Condvar,
    }

    fn heaps() -> &'static Heaps {
        static HEAPS: OnceLock<Heaps> = OnceLock::new();
        HEAPS.get_or_init(|| {
            std::thread::Builder::new()
                .name("pie-heaps".into())
                .spawn(make_ahead)
                .expect("the heaps' thread starts");
            Heaps {
                heaps: Mutex::default(),
                wanted: Mutex::default(),
                wake: Condvar::new(),
            }
        })
    }

    fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex.lock().expect("mutex lock poisoned")
    }

    /// The heaps' thread: makes the chunks asked for, one heap at a time,
    /// holding no lock while the driver does.
    fn make_ahead() {
        let heaps = heaps();
        loop {
            let key = {
                let mut wanted = locked(&heaps.wanted);
                loop {
                    if let Some(key) = wanted.pop() {
                        break key;
                    }
                    wanted = heaps.wake.wait(wanted).expect("wake lock poisoned");
                }
            };
            let made = (key.ordinal < 0 || crate::device::ctx::bind_thread(key.ordinal).is_ok())
                .then(|| key.make_chunk(key.kind.chunk()).ok())
                .flatten()
                .map(|cell| (cell, key.kind.chunk()));
            let mut all = locked(&heaps.heaps);
            let heap = all.entry(key).or_default();
            heap.making = false;
            heap.ready.extend(made);
        }
    }

    /// Asks the heaps' thread for a chunk ahead for `key`, unless one is
    /// ready or under way.
    fn ask_ahead(heap: &mut Heap, key: Key) {
        if heap.ready.is_empty() && !heap.making {
            heap.making = true;
            let heaps = heaps();
            locked(&heaps.wanted).push(key);
            heaps.wake.notify_one();
        }
    }

    /// The class `bytes` of `kind` is carved in, if any.
    #[must_use]
    pub fn class_of(kind: Kind, bytes: usize) -> Option<usize> {
        (bytes <= kind.largest()).then(|| {
            (bytes.max(kind.smallest()).next_power_of_two() / kind.smallest()).trailing_zeros()
                as usize
        })
    }

    /// Makes `kind`'s first chunk on this thread's device, at load, and asks
    /// for the one after it.
    pub fn reserve(kind: Kind) -> Result<()> {
        let Some(key) = Key::of(kind) else {
            return Ok(());
        };
        let fresh = {
            let all = locked(&heaps().heaps);
            all.get(&key).is_none_or(|heap| heap.current.is_none())
        };
        if fresh {
            let bytes = kind.chunk();
            let chunk = key.make_chunk(bytes)?;
            let mut all = locked(&heaps().heaps);
            let heap = all.entry(key).or_default();
            if heap.current.is_none() {
                heap.current = Some(Chunk {
                    at: chunk,
                    used: 0,
                    bytes,
                });
            } else {
                heap.ready.push((chunk, bytes));
            }
            ask_ahead(heap, key);
        }
        Ok(())
    }

    /// `count` summed over every kept cell's fence.
    pub(super) fn fences(count: &dyn Fn(&Fence) -> usize) -> usize {
        let all = locked(&heaps().heaps);
        all.values()
            .flat_map(|heap| heap.free.iter().flatten())
            .map(|kept| count(&kept.fence))
            .sum()
    }

    /// Drops `stream`'s marks from every kept cell's fence
    /// ([`super::retire_stream`]).
    pub(super) fn forget_stream(stream: usize) {
        let mut all = locked(&heaps().heaps);
        for heap in all.values_mut() {
            for kept in heap.free.iter_mut().flatten() {
                kept.fence.forget_stream(stream);
            }
        }
    }

    /// A cell of `class`: a freed one whose last reader has landed, else one
    /// carved from the current chunk or one made ahead. Only a heap that
    /// has run dry makes a chunk here, holding no lock while it does.
    pub fn take(kind: Kind, class: usize) -> Result<Cell> {
        let bytes = kind.class_bytes(class);
        let key = Key::of(kind).ok_or_else(|| {
            crate::error::Fault::Unsupported(
                "a heap cell asked for with no device bound".to_string(),
            )
        })?;
        loop {
            {
                let mut all = locked(&heaps().heaps);
                let heap = all.entry(key).or_default();
                if heap.free.len() < kind.classes() {
                    heap.free.resize_with(kind.classes(), Vec::new);
                }
                let list = &mut heap.free[class];
                if let Some(at) = list.iter().position(|kept| kept.fence.landed()) {
                    return Ok(list.swap_remove(at).cell);
                }
                let fits =
                    |chunk: &Chunk| chunk.used.next_multiple_of(bytes) + bytes <= chunk.bytes;
                if !heap.current.as_ref().is_some_and(fits)
                    && let Some((next, size)) = heap.ready.pop()
                {
                    heap.current = Some(Chunk {
                        at: next,
                        used: 0,
                        bytes: size,
                    });
                    ask_ahead(heap, key);
                }
                if let Some(chunk) = heap.current.as_mut().filter(|chunk| fits(chunk)) {
                    let offset = chunk.used.next_multiple_of(bytes);
                    chunk.used = offset + bytes;
                    return Ok(chunk.at.at(offset));
                }
            }
            // Dry: the chunk is made here, outside the lock, and carved from
            // on the next pass.
            let chunk = key.make_chunk_for(bytes)?;
            let mut all = locked(&heaps().heaps);
            all.entry(key).or_default().ready.push(chunk);
        }
    }

    /// Gives `cell` of `class` back, behind a fence on the compute streams;
    /// a cell no fence can guard is not handed out again.
    pub fn give(kind: Kind, class: usize, cell: Cell) {
        let key = Key {
            kind,
            ordinal: cell.ordinal,
        };
        // Pinned memory is read by whichever device this thread serves.
        let device = match kind {
            Kind::Pinned => current_device(),
        };
        let Some(fence) = device.map_or(Some(Fence(Vec::new())), Fence::record) else {
            return;
        };
        let mut all = locked(&heaps().heaps);
        let heap = all.entry(key).or_default();
        if heap.free.len() < kind.classes() {
            heap.free.resize_with(kind.classes(), Vec::new);
        }
        heap.free[class].push(Kept { cell, fence });
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_size_is_carved_in_the_least_power_of_two_that_holds_it() {
            for kind in [Kind::Pinned] {
                assert_eq!(class_of(kind, 1), Some(0));
                assert_eq!(class_of(kind, kind.smallest()), Some(0));
                assert_eq!(class_of(kind, kind.smallest() + 1), Some(1));
                assert_eq!(kind.class_bytes(class_of(kind, 3000).unwrap()), 4096);
                assert_eq!(class_of(kind, kind.largest()), Some(kind.classes() - 1));
                assert_eq!(class_of(kind, kind.largest() + 1), None);
            }
        }

        #[test]
        fn a_card_short_of_a_whole_chunk_still_carves_a_small_cell() {
            let cell = Cell {
                host: 0,
                device: 1,
                ordinal: 0,
            };
            let room = 24 << 20;
            let (mut asked, mut trims) = (Vec::new(), 0);
            let made = chunk_within(
                256 << 20,
                1 << 20,
                |size| {
                    asked.push(size >> 20);
                    if size <= room {
                        Ok(cell)
                    } else {
                        Err(crate::error::Fault::OutOfMemory {
                            need: size as u64,
                            have: room as u64,
                        })
                    }
                },
                || trims += 1,
            );
            assert_eq!(made.unwrap().1, 16 << 20);
            assert_eq!(asked, [256, 256, 128, 64, 32, 16]);
            assert_eq!(trims, 1, "what is kept idle goes back once, first");
            let short = chunk_within(
                256 << 20,
                64 << 20,
                |size| {
                    Err(crate::error::Fault::OutOfMemory {
                        need: size as u64,
                        have: 0,
                    })
                },
                || {},
            );
            assert!(matches!(
                short,
                Err(crate::error::Fault::OutOfMemory { need, .. }) if need == 64 << 20
            ));
        }
    }
}
