//! Device memory kept behind a fence on the compute streams and handed out
//! again once it lands, rather than given back to the driver and asked for
//! anew.
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicPtr, Ordering};

use crate::device::rt;

/// `cudaEventDisableTiming`, as HIP's is too.
pub(crate) const DISABLE_TIMING: u32 = 2;

const ORDINALS: usize = 64;

static STREAMS: [AtomicPtr<core::ffi::c_void>; ORDINALS] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; ORDINALS];

/// The live contexts' compute streams, by ordinal, in adoption order: a
/// stream is recorded on only while this is read-held, and retired (and
/// destroyed) only while it is write-held, so no event is recorded on a
/// destroyed stream, whichever context of the process goes first.
static LIVE: std::sync::RwLock<Vec<(i32, usize)>> = std::sync::RwLock::new(Vec::new());

/// A context's compute stream, every fire of `ordinal` it runs launching on
/// it; the latest live one is where kept memory is zeroed for reuse.
pub fn adopt_stream(ordinal: i32, stream: *mut core::ffi::c_void) {
    let mut live = LIVE.write().expect("live lock poisoned");
    live.push((ordinal, stream as usize));
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
        .position(|&entry| entry == (ordinal, stream as usize))
    {
        live.remove(at);
    }
    let next = live
        .iter()
        .rev()
        .find(|entry| entry.0 == ordinal)
        .map_or(core::ptr::null_mut(), |entry| {
            entry.1 as *mut core::ffi::c_void
        });
    if let Some(slot) = usize::try_from(ordinal).ok().and_then(|at| STREAMS.get(at)) {
        slot.store(next, Ordering::Release);
    }
    destroy();
}

/// The device `ptr`, an allocation of device memory, lives on.
fn device_of(ptr: u64) -> Option<i32> {
    // SAFETY: a zeroed attribute record the call fills.
    let mut attrs: rt::cudaPointerAttributes = unsafe { core::mem::zeroed() };
    // SAFETY: `attrs` is a live local the call writes; `ptr` is only looked up.
    let code =
        unsafe { rt::cudaPointerGetAttributes(&raw mut attrs, ptr as *const core::ffi::c_void) };
    (code == rt::cudaError::cudaSuccess && attrs.type_ == rt::cudaMemoryType::cudaMemoryTypeDevice)
        .then_some(attrs.device)
}

pub(crate) fn current_device() -> Option<i32> {
    let mut device = 0i32;
    // SAFETY: writes one int.
    let code = unsafe { rt::cudaGetDevice(&raw mut device) };
    (code == rt::cudaError::cudaSuccess).then_some(device)
}

fn stream_of(device: i32) -> Option<*mut core::ffi::c_void> {
    let stream = STREAMS
        .get(usize::try_from(device).ok()?)?
        .load(Ordering::Acquire);
    (!stream.is_null()).then_some(stream)
}

/// Whether `device` has an adopted compute stream.
pub(crate) fn adopted(device: i32) -> bool {
    stream_of(device).is_some()
}

/// `use_stream` on `device`'s latest adopted compute stream, kept live for
/// the call; none if it has none.
pub(crate) fn with_stream<R>(
    device: i32,
    use_stream: impl FnOnce(*mut core::ffi::c_void) -> R,
) -> Option<R> {
    let _live = LIVE.read().expect("live lock poisoned");
    stream_of(device).map(use_stream)
}

/// An event on every live compute stream of a device, recorded at once:
/// whatever any of them had enqueued before is done once all have landed.
/// One context's stream alone would not do while another context on the
/// device may still read what is kept.
pub(crate) struct Fence(Vec<rt::cudaEvent_t>);

// SAFETY: the events are process-wide device objects, each destroyed once.
unsafe impl Send for Fence {}

impl Fence {
    /// `None` where a stream would not record; an empty fence where the
    /// device has no live stream, so nothing of it is airborne.
    pub(crate) fn record(device: i32) -> Option<Fence> {
        let live = LIVE.read().expect("live lock poisoned");
        let mut fence = Fence(Vec::new());
        for &(_, stream) in live.iter().filter(|(ordinal, _)| *ordinal == device) {
            let mut event: rt::cudaEvent_t = core::ptr::null_mut();
            // SAFETY: creates a fresh event and records it on a stream the
            // read-held list keeps live.
            let recorded = unsafe {
                rt::cudaEventCreateWithFlags(&raw mut event, DISABLE_TIMING)
                    == rt::cudaError::cudaSuccess
                    && {
                        fence.0.push(event);
                        rt::cudaEventRecord(event, stream as *mut core::ffi::c_void as _)
                            == rt::cudaError::cudaSuccess
                    }
            };
            if !recorded {
                return None;
            }
        }
        Some(fence)
    }

    pub(crate) fn landed(&self) -> bool {
        self.0.iter().all(|&event| {
            // SAFETY: each event is this fence's own.
            unsafe { rt::cudaEventQuery(event) == rt::cudaError::cudaSuccess }
        })
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        for &event in &self.0 {
            // SAFETY: each event is this fence's own, destroyed once.
            unsafe { rt::cudaEventDestroy(event) };
        }
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

    use super::{Fence, adopted, current_device, with_stream};
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

    struct Entry {
        held: Held,
        fence: Fence,
    }

    static POOL: Mutex<Option<HashMap<PoolKey, Vec<Entry>>>> = Mutex::new(None);

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
    /// memory zeroed (the device cells on the compute stream, ahead of any
    /// fire that reads them).
    pub fn take(key: PoolKey) -> Option<Held> {
        if !adopted(key.device) {
            return None;
        }
        let mut pool = POOL.lock().expect("pool lock poisoned");
        let list = pool.as_mut()?.get_mut(&key)?;
        let at = list.iter().position(|entry| entry.fence.landed())?;
        let entry = list.swap_remove(at);
        drop(pool);
        let (words, mirror, cells) = entry.held;
        words.zero();
        mirror.zero();
        if let Some(cells) = &cells {
            // SAFETY: the buffer is the entry's, no airborne fire reads it,
            // and the memset is ordered before every later fire.
            let code = with_stream(key.device, |stream| unsafe {
                rt::cudaMemsetAsync(
                    cells.ptr() as *mut core::ffi::c_void,
                    0,
                    cells.bytes(),
                    stream.cast(),
                )
            });
            if code != Some(rt::cudaError::cudaSuccess) {
                return None;
            }
        }
        Some((words, mirror, cells))
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
        if list.len() >= PER_KEY {
            return;
        }
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
pub(crate) mod recycle {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{Fence, current_device, device_of};
    use crate::device::rt;
    use crate::error::Result;

    /// The bytes kept across every size at most; an allocation the device
    /// refuses frees them all first, so this bounds only what an idle load
    /// sits on.
    const BUDGET: usize = 8 << 30;

    struct Kept {
        ptr: u64,
        fence: Fence,
    }

    /// Kept buffers by (device, bytes).
    type Shelf = Option<HashMap<(i32, usize), Vec<Kept>>>;

    static KEPT: Mutex<Shelf> = Mutex::new(None);
    static HELD: AtomicUsize = AtomicUsize::new(0);

    fn kept() -> std::sync::MutexGuard<'static, Shelf> {
        KEPT.lock().expect("kept lock poisoned")
    }

    /// `bytes` of device memory: a kept buffer whose last reader has
    /// landed, else a fresh one; on a refusal everything kept is freed and
    /// the allocation is tried once more.
    pub(crate) fn alloc(bytes: usize) -> Result<u64> {
        if let Some(device) = current_device() {
            let mut kept = kept();
            if let Some(list) = kept.as_mut().and_then(|map| map.get_mut(&(device, bytes)))
                && let Some(at) = list.iter().position(|entry| entry.fence.landed())
            {
                let entry = list.swap_remove(at);
                drop(kept);
                HELD.fetch_sub(bytes, Ordering::Relaxed);
                return Ok(entry.ptr);
            }
        }
        let fresh = |bytes: usize| {
            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `base` is a live local the call fills.
            let code = unsafe { rt::cudaMalloc(&raw mut base, bytes) };
            crate::device::ctx::check("cudaMalloc", code).map(|()| base as u64)
        };
        match fresh(bytes) {
            Ok(ptr) => Ok(ptr),
            Err(_) if drain() => {
                fresh(bytes).map_err(|fault| crate::device::alloc::out_of_room(fault, bytes))
            }
            Err(fault) => Err(crate::device::alloc::out_of_room(fault, bytes)),
        }
    }

    /// Keeps `ptr` for its size behind a fence on its device's compute
    /// streams, or frees it when the budget is spent or the fence will not
    /// record. Its device is the one it was made on, whichever this thread
    /// has bound: a group's leader drops every rank's buffers.
    pub(crate) fn release(ptr: u64, bytes: usize) {
        if HELD.load(Ordering::Relaxed) + bytes <= BUDGET
            && let Some(device) = device_of(ptr)
            && let Some(fence) = Fence::record(device)
        {
            HELD.fetch_add(bytes, Ordering::Relaxed);
            kept()
                .get_or_insert_with(HashMap::new)
                .entry((device, bytes))
                .or_default()
                .push(Kept { ptr, fence });
            return;
        }
        // SAFETY: the pointer came from `alloc` and is freed exactly once.
        let _ = unsafe { rt::cudaFree(ptr as *mut core::ffi::c_void) };
    }

    /// Frees every kept buffer whose last reader has landed; the bytes
    /// given back. A buffer kept is held for the next of its size, and is
    /// idle until then: at a load, and once a shell is down, nothing is
    /// served that would ask for it before the driver could hand it out
    /// again, so the device's word on what is free is the process's own.
    /// Each free is of memory no frame reads, made where the engine's
    /// thread already allocates and frees: never mid-step, where a free
    /// could wait on a peer spinning for this rank.
    pub(crate) fn trim() -> usize {
        let idle: Vec<(u64, usize)> = {
            let mut kept = kept();
            let Some(map) = kept.as_mut() else {
                return 0;
            };
            let mut idle = Vec::new();
            for (&(_, bytes), list) in map.iter_mut() {
                list.retain(|entry| {
                    let landed = entry.fence.landed();
                    if landed {
                        idle.push((entry.ptr, bytes));
                    }
                    !landed
                });
            }
            map.retain(|_, list| !list.is_empty());
            idle
        };
        let mut freed = 0;
        for (ptr, bytes) in idle {
            HELD.fetch_sub(bytes, Ordering::Relaxed);
            freed += bytes;
            // SAFETY: the pointer came from `alloc` and left the shelf
            // above, so it is freed exactly once; its fence has landed, so
            // no reader is left.
            unsafe {
                let _ = rt::cudaFree(ptr as *mut core::ffi::c_void);
            }
        }
        freed
    }

    /// The bytes kept now, across every device.
    pub(crate) fn kept_bytes() -> usize {
        HELD.load(Ordering::Relaxed)
    }

    /// Frees every kept buffer; whether there was any.
    fn drain() -> bool {
        let Some(map) = kept().take() else {
            return false;
        };
        let mut any = false;
        for entry in map.into_values().flatten() {
            any = true;
            // SAFETY: each pointer is its entry's own; `cudaFree` waits for
            // the device, so no reader is left.
            unsafe {
                let _ = rt::cudaFree(entry.ptr as *mut core::ffi::c_void);
            }
        }
        HELD.store(0, Ordering::Relaxed);
        any
    }
}

/// Gives the driver back the device memory the process keeps idle
/// ([`recycle`]'s dropped buffers whose last reader has landed); the bytes
/// freed. A load calls it before it measures the card, and a shell once it
/// is down, so a model loaded after another was dropped fits as it would in
/// a fresh process.
pub fn release_idle() -> usize {
    recycle::trim()
}

/// The device bytes the process keeps from dropped buffers, for the next
/// of their size ([`recycle`]).
#[must_use]
pub fn kept_bytes() -> usize {
    recycle::kept_bytes()
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
        Device,
    }

    impl Kind {
        const fn smallest(self) -> usize {
            match self {
                Kind::Pinned => 64,
                Kind::Device => 256,
            }
        }

        const fn largest(self) -> usize {
            match self {
                Kind::Pinned => 1 << 20,
                Kind::Device => 64 << 20,
            }
        }

        /// What one chunk holds.
        const fn chunk(self) -> usize {
            match self {
                Kind::Pinned => 64 << 20,
                Kind::Device => 256 << 20,
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
                    Kind::Device => current_device()?,
                },
            })
        }

        /// One more chunk; the calling thread is bound to the heap's device.
        fn make_chunk(self) -> Result<Cell> {
            let bytes = self.kind.chunk();
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
                Kind::Device => {
                    let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
                    // SAFETY: `device` is a live local the call fills; the
                    // chunk is kept for the life of the process.
                    let made = unsafe {
                        crate::device::ctx::check(
                            "cudaMalloc",
                            rt::cudaMalloc(&raw mut device, bytes),
                        )
                    };
                    made.map_err(|fault| crate::device::alloc::out_of_room(fault, bytes))?;
                    Ok(Cell {
                        host: 0,
                        device: device as u64,
                        ordinal: self.ordinal,
                    })
                }
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
    }

    struct Kept {
        cell: Cell,
        fence: Fence,
    }

    #[derive(Default)]
    struct Heap {
        /// The chunk cells are carved from.
        current: Option<Chunk>,
        /// Chunks made ahead, not yet carved from.
        ready: Vec<Cell>,
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
                .then(|| key.make_chunk().ok())
                .flatten();
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
            let chunk = key.make_chunk()?;
            let mut all = locked(&heaps().heaps);
            let heap = all.entry(key).or_default();
            if heap.current.is_none() {
                heap.current = Some(Chunk { at: chunk, used: 0 });
            } else {
                heap.ready.push(chunk);
            }
            ask_ahead(heap, key);
        }
        Ok(())
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
                    |chunk: &Chunk| chunk.used.next_multiple_of(bytes) + bytes <= kind.chunk();
                if !heap.current.as_ref().is_some_and(fits)
                    && let Some(next) = heap.ready.pop()
                {
                    heap.current = Some(Chunk { at: next, used: 0 });
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
            let chunk = key.make_chunk()?;
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
            Kind::Device => Some(cell.ordinal),
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
            for kind in [Kind::Pinned, Kind::Device] {
                assert_eq!(class_of(kind, 1), Some(0));
                assert_eq!(class_of(kind, kind.smallest()), Some(0));
                assert_eq!(class_of(kind, kind.smallest() + 1), Some(1));
                assert_eq!(kind.class_bytes(class_of(kind, 3000).unwrap()), 4096);
                assert_eq!(class_of(kind, kind.largest()), Some(kind.classes() - 1));
                assert_eq!(class_of(kind, kind.largest() + 1), None);
            }
        }
    }
}
