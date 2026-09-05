//! The checkpoint, resident: one device allocation, one row per
//! `Trace::params`, plus the adapter banks in the same allocation and table.
//! Banks live here, not in `store/`: they share `Def::Weight`'s one
//! resolution path and are written between fires by `register_adapter`,
//! not by launches in the fire path. `trace.params[i].name` must equal the
//! contract's published name, or resolution fails at first fire with
//! [`WeightTable`]'s `None`.

/// The device load arena and its four transforms, gated on a chosen runtime.
#[cfg(feature = "cuda")]
pub mod arena;

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::file::read::parse_metadata;
use checkpoint::file::zt;
use checkpoint::contract::ModelContract;
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::plan::{LoadPlan, StorageTarget, compile, compile_streaming};
use checkpoint::types::{ScaleForm, TensorId};
use kernels_cuda::Tensor;
use kernels_cuda::linear::moe::GroupSeat;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::experts::Attachments;
use crate::run::{WeightRow, WeightTable};

/// What a matrix operand wants under cuBLAS, and what `cudaMalloc` itself
/// guarantees — the same alignment the loader's `StorageTarget` states.
pub(crate) const ALIGN: u64 = 256;

/// One plane of a registered adapter. `bytes` must be exactly one
/// full-capacity slot of `bank`, zero-padded by the caller if the adapter
/// was trained at a lower rank; a short plane errors as `Fault::Adapter`.
#[derive(Debug, Clone, Copy)]
pub struct AdapterPlane<'a> {
    /// The bank param this plane fills, as `Trace::params` names it.
    pub bank: &'a str,
    /// One slot's worth of bytes.
    pub bytes: &'a [u8],
}

/// Every weight this model needs, on the device — the checkpoint's and the
/// banks'.
#[derive(Debug)]
pub struct Weights {
    store: Buffer,
    table: WeightTable,
    /// Keyed by bank name, built at load off `ParamSource::Registered`.
    banks: BTreeMap<String, Bank>,
    /// The routed-expert tier, or `None` when the device budget covers the
    /// whole table.
    experts: Option<crate::experts::Tier>,
    /// `true` means the host-side transform pipeline never ran for this load.
    from_cache: bool,
    /// `Some` means some spilled dense planes are read out of device slots
    /// that rotate during the fire; weight-table addresses never move.
    rotor: Option<crate::rotate::Rotor>,
}

/// One declared adapter bank: where its slots are and how big they are.
#[derive(Debug, Clone, Copy)]
struct Bank {
    offset: u64,
    adapters: u32,
    slot: u64,
    /// One slot's rectangle, past the leading adapters axis — `[rank, in]`
    /// for an `A`, `[out, rank]` for a `B`. Needed by the shared-blob
    /// resolver only.
    rows: u64,
    cols: u64,
    elem: u64,
}

/// One bank, as [`crate::blob`]'s shared-adapter resolver reads it: a
/// flattened [`Bank`], since the resolver lives in another module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankSeat {
    /// The param's own name, which is what a registration names.
    pub name: String,
    /// How many adapters it seats.
    pub adapters: u32,
    /// One adapter's bytes.
    pub slot: u64,
    /// The leading axis of one slot.
    pub rows: u64,
    /// The trailing axes of one slot, multiplied out.
    pub cols: u64,
    /// One element in bytes.
    pub elem: u64,
}

/// The banks a plan declares, read off `ParamSource::Registered`. A bank's
/// capacity is its param's leading axis; one adapter's slot is everything
/// after it. `A` and `B` are independent banks — the op pairs them, not this.
fn banks(trace: &Trace, places: &[Place]) -> BTreeMap<String, Bank> {
    trace.params
        .iter()
        .zip(places)
        .filter(|(param, _)| param.source == ParamSource::Registered)
        .map(|(param, place)| {
            let adapters = u32::try_from(param.shape.first().copied().unwrap_or(0))
                .unwrap_or(u32::MAX);
            let slot = if adapters == 0 {
                0
            } else {
                place.bytes / u64::from(adapters)
            };
            // Slot rectangle: the param's shape with the adapters axis cut off.
            let (rows, cols) = rectangle(param.shape.get(1..).unwrap_or(&[]));
            (
                param.name.clone(),
                Bank {
                    offset: place.offset,
                    adapters,
                    slot,
                    rows,
                    cols,
                    elem: model_compiler::arena::elem_bytes(param.dtype).unwrap_or(0),
                },
            )
        })
        .collect()
}

/// How many device bytes this plan's weight table demands resident, before a
/// byte is allocated. A pure function of the trace, so an unadmittable
/// [`Residency`](engine::load::Residency) budget is refused early.
/// # Errors
/// [`Fault::Param`] for a param whose dtype has no element size.
pub fn device_demand(trace: &Trace) -> Result<u64> {
    let places = places(trace, &crate::experts::Plan::default())?;
    Ok(places.last().map_or(0, |place| place.offset + place.reserved))
}

/// Every param's plane bytes, unaligned and unreduced. Shared by [`places`]
/// and [`experts::Plan::of`](crate::experts::Plan::of) so both agree on a
/// bank's size. Packed dtypes (mxfp4, MLX affine) are sized from their
/// already-packed shape, not through `elem_bytes`.
/// # Errors
/// [`Fault::Param`] for a param declared in a storage element with no byte size.
pub(crate) fn plane_bytes(trace: &Trace) -> Result<Vec<u64>> {
    trace
        .params
        .iter()
        .map(|param| {
            let (rows, width) = rectangle(&param.shape);
            Ok(match param.dtype {
                // The shape already folds a 32-code block into sixteen bytes.
                Dtype::Mxfp4 => rows.saturating_mul(width),
                // Nibble element: two codes to a byte, odd row rounds up.
                Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => {
                    rows.saturating_mul(width).div_ceil(2)
                }
                // Two-bit codes, four to a byte; group size is in the companion planes.
                Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => {
                    rows.saturating_mul(width).div_ceil(4)
                }
                // One whole byte a code, so nothing rounds.
                Dtype::U8g64 => rows.saturating_mul(width),
                // Braided plane: shape is already `[n, Dtype::row_bytes(k)]`.
                Dtype::U2g16k
                | Dtype::I3g16k
                | Dtype::U4g32k
                | Dtype::U5g32k
                | Dtype::I6g16k => rows.saturating_mul(width),
                other => {
                    let element =
                        model_compiler::arena::elem_bytes(other).ok_or_else(|| Fault::Param {
                            name: param.name.clone(),
                            why: "is declared in a packed storage element that has no \
                                  element size",
                        })?;
                    rows.saturating_mul(width).saturating_mul(element)
                }
            })
        })
        .collect()
}

/// Everything a load can be planned against before a byte of it is read.
/// Both fields come off the same `LoadPlan` compile.
#[derive(Debug, Clone)]
pub struct Prospect {
    /// The split-plane pairings: which other params move when a packed bank
    /// moves. What [`experts::Plan::of`](crate::experts::Plan::of) budgets
    /// groups with.
    pub planes: Attachments,
    /// The priority ranking this trace declares, before any budget cuts it.
    /// What [`experts::Plan::cut`](crate::experts::Plan::cut) turns into a
    /// residency.
    pub ranking: crate::experts::Ranking,
}

/// A routed bank's other device planes and the T2 source's name, so a
/// residency decision can be made before [`Weights::resident`] reserves the
/// store. Reads no tensor bytes.
/// # Errors
/// [`Fault::Load`] for a checkpoint the contract does not fit,
/// [`Fault::Param`] for an attachment this plan cannot resolve.
pub fn prospect(
    trace: &Trace,
    contract: &ModelContract,
    path: &Path,
    target: StorageTarget,
) -> Result<Prospect> {
    let metadata = if path.is_dir() {
        parse_metadata(path)?
    } else {
        zt::parse(path)?
    };
    let landing = compile(&metadata, contract, target)?;
    let index: BTreeMap<&str, usize> = trace
        .params
        .iter()
        .enumerate()
        .map(|(at, param)| (param.name.as_str(), at))
        .collect();
    let planes = attachments(&landing, &index)?;
    let ranking = crate::experts::Ranking::of(trace, &planes)?;
    Ok(Prospect {
        ranking,
        planes,
    })
}

/// The split-plane pairings, as a residency decision wants them: which other
/// params move when a packed bank moves, keyed by the bank's own row.
/// # Errors
/// [`pairings`]', verbatim.
fn attachments(landing: &LoadPlan, index: &BTreeMap<&str, usize>) -> Result<Attachments> {
    let mut planes = Attachments::new();
    for (name, pairing) in pairings(landing, index)? {
        let Some(&at) = index.get(name) else {
            continue;
        };
        let mut companions = vec![pairing.scales];
        companions.extend(pairing.biases);
        planes.insert(at, companions);
    }
    Ok(planes)
}

/// Restores every device image from the checkpoint itself: T1 into the
/// pinned allocation, T0 as staged transfers, both hash-verified as they
/// arrive.
/// # Errors
/// [`Rotten::Bytes`] for a file that does not answer for a plane or whose
/// blocks do not hash, [`Rotten::Machine`] for a device or disk that did
/// not. The file is left where it is either way.
fn restore_from_checkpoint(
    serving: &crate::checkpoint_serving::Serving,
    trace: &Trace,
    plan: &crate::experts::Plan,
    places: &[Place],
    store: &mut Buffer,
    tier: Option<&mut crate::experts::Tier>,
) -> std::result::Result<(), Rotten> {
    // A deferred seat's T1 is a verify, not a copy: kernels fault its pages
    // in themselves. `tier` is `None` for a resident plan (no T1 arm).
    let layout = tier.as_ref().map(|tier| tier.plan().host_layout()).unwrap_or_default();
    let seated = tier.as_ref().is_some_and(|tier| tier.deferred_image().is_some());
    let refill = match seated || layout.is_empty() {
        true => None,
        false => Some(serving.refill(&layout).map_err(Rotten::Bytes)?),
    };
    let pinned_params: Vec<u32> = layout
        .iter()
        .map(|(param, _, _, _)| u32::try_from(*param).unwrap_or(u32::MAX))
        .collect();

    // T0: one transfer per image the store holds, at the store's own offsets.
    let base = store.at(0).map_err(|why| Rotten::Machine(format!("{why}")))?;
    let mut transfers = Vec::with_capacity(places.len());
    let mut device_params = Vec::with_capacity(places.len());
    for (param, place) in places.iter().enumerate() {
        // A registered plane is zeroed at load and filled by `register_adapter`.
        if place.reserved == 0
            || trace.params.get(param).map(|p| p.source) != Some(ParamSource::Checkpoint)
        {
            continue;
        }
        // A streamed bank's slab is not pumped: `Tier::land` fills its
        // resident slots from the pinned copy, so pumping would overwrite it.
        if plan.resident(param).is_some() {
            continue;
        }
        let id = u32::try_from(param).unwrap_or(u32::MAX);
        let len = place.bytes;
        let Some(src) = serving
            .plane(id)
            .filter(|src| src.len() as u64 == len && len <= place.reserved)
        else {
            let plane = serving
                .name(id)
                .map_or_else(|| format!("param {param}"), |name| format!("`{name}`"));
            return Err(Rotten::Bytes(format!(
                "carries no {len}-byte image for {plane}, which this trace puts in the \
                 device store"
            )));
        };
        // The one bounds check on this path: this arm never reaches `Buffer::write`.
        store
            .at(place.offset.saturating_add(len))
            .map_err(|why| Rotten::Machine(format!("an image does not fit the store: {why}")))?;
        transfers.push(crate::staged_h2d::Transfer {
            dst: base + place.offset,
            src: src.as_ptr(),
            len,
        });
        device_params.push(id);
    }

    /// The pinned allocation's base, as something a scope thread may carry.
    struct Into(*mut u8);
    // SAFETY: the allocation is the tier's, handed to nobody else, and the
    // thread it moves into is its sole writer for as long as the scope is open.
    unsafe impl Send for Into {}

    // T2's spilled planes are hashed too: a mapped plane has no first-touch
    // hook, so without this a flipped bit in it would boot clean.
    let mapped: Vec<u32> = plan
        .mapped_layout()
        .iter()
        .map(|(param, _, _, _)| u32::try_from(*param).unwrap_or(u32::MAX))
        .collect();
    let mut hashed = device_params;
    hashed.extend(mapped);
    let (pumped, pinned) = (transfers.len(), pinned_params.len());
    // Null when there is no tier; never dereferenced there since `refill` is `None` too.
    let into = Into(tier.as_ref().map_or(std::ptr::null_mut(), |tier| tier.host().host()));
    let (read, moved, verified) = std::thread::scope(|scope| {
        // SAFETY: `into` is the tier's own uninitialized allocation, which
        // `host_layout` tiles exactly, and no other reader names it yet.
        let reading =
            scope.spawn(move || match &refill {
                // A deferred seat: images stay where they lie, only the claim is checked.
                None => serving.verify_planes(&pinned_params),
                // Move the whole value: edition 2021 would capture `into.0`, not `Send`.
                Some(refill) => {
                    let into = into;
                    unsafe { crate::checkpoint_serving::read_into(refill, into.0) }
                }
            });
        let (moved, verified) = match (transfers.is_empty(), hashed.is_empty()) {
            (true, true) => (Ok(()), Ok(Ok(()))),
            (true, false) => (Ok(()), Ok(serving.verify_planes(&hashed))),
            (false, _) => {
                let mut lanes = match crate::staged_h2d::Lanes::standard() {
                    Ok(lanes) => lanes,
                    Err(why) => return (reading.join(), Err(why), Ok(Ok(()))),
                };
                // Device digests run beside their own lanes' mapping.
                std::thread::scope(|inner| {
                    let hashing = inner.spawn(|| serving.verify_planes(&hashed));
                    let moved = lanes.pump(&transfers);
                    (moved, hashing.join())
                })
            }
        };
        (reading.join(), moved, verified)
    });

    // Check the host arm first: its destination skipped a memset.
    match read {
        Ok(Ok(())) => {}
        Ok(Err(why)) => return Err(Rotten::Bytes(why)),
        Err(_) => return Err(Rotten::Machine("a host reader panicked".to_string())),
    }
    moved.map_err(|why| Rotten::Machine(format!("staged upload failed: {why}")))?;
    match verified {
        Ok(Err(why)) => return Err(Rotten::Bytes(why)),
        Err(_) => return Err(Rotten::Machine("a digest worker panicked".to_string())),
        Ok(Ok(())) => {}
    }
    eprintln!(
        "engine-cuda: this boot read its whole image out of {:?} — {} device image(s) \
         pumped and {} T1 plane(s) {}, so the executor never ran",
        serving.path(),
        pumped,
        pinned,
        if seated { "verified where they lie" } else { "copied and verified" },
    );
    Ok(())
}

/// The artifact to serve T1 from during a warm boot, or `None` to page-lock
/// the image up front. Deferring verifies T1 planes where they lie instead of
/// copying them, at the cost of page faults until the background fill lands.
/// Requires a warm artifact, `host_image() > 0`, and `pageable_access`.
fn defer_tiers(
    serving: Option<&crate::checkpoint_serving::Serving>,
    plan: &crate::experts::Plan,
) -> Option<crate::checkpoint_serving::Serving> {
    let serving = serving?;
    if plan.host_image() == 0 || !crate::experts::pageable_access() {
        return None;
    }
    // A spill and a deferred seat ask the artifact for different sets; the
    // seat's needs can be missing even when a spill's are met. Fall back.
    if let Err(why) = serving.covers(&plan.host_layout()) {
        eprintln!(
            "engine-cuda: the tier is not deferred — {why}, so this boot builds its \
             page-locked image the eager way; the load is unaffected"
        );
        return None;
    }
    Some(serving.clone())
}

/// Why a warm streamed boot did not get its images. The two variants exist
/// so the operator is told whether to re-import (bad bytes) or check the
/// machine (a device or disk that stopped answering); neither deletes the file.
enum Rotten {
    Bytes(String),
    Machine(String),
}

/// Arms the background fill behind a deferred seat: spawns the thread and
/// hands it its end of the channel. A fill that cannot be armed just leaves
/// the seat serving out of the mapping for its whole life.
fn arm_refill(tier: &mut crate::experts::Tier) {
    // The tier's own mapping, already verified by the caller.
    let Some(image) = tier.deferred_image() else {
        return;
    };
    // Built before the spawn so nothing crossing it aliases the tier or artifact.
    let refill = match image.refill(&tier.plan().host_layout()) {
        Ok(refill) => refill,
        Err(why) => {
            eprintln!(
                "engine-cuda: the deferred tier's fill cannot be described ({why}); it will \
                 serve out of the artifact for the life of this load"
            );
            return;
        }
    };
    let bytes = usize::try_from(tier.plan().host_image()).unwrap_or(usize::MAX);
    // `cudaSetDevice` is per-thread, so the ordinal is read here and carried.
    let ordinal = match crate::device::ctx::current() {
        Ok(ordinal) => ordinal,
        Err(why) => {
            eprintln!(
                "engine-cuda: the deferred tier cannot name its device ({why}); it will \
                 serve out of {:?} for the life of this load",
                refill.path,
            );
            return;
        }
    };
    let (send, filled) = std::sync::mpsc::channel();
    // Logged even though every other line here is a refusal: until the fill
    // lands, every T1 read is an NVMe page fault over HMM, and a log that
    // never says the road was taken cannot show it.
    let objects = refill.landings.len();
    let path = refill.path.clone();
    match std::thread::Builder::new()
        .name("pie-tier-refill".to_string())
        .spawn(move || refill_from(&refill, bytes, ordinal, &send))
    {
        Ok(filling) => {
            eprintln!(
                "engine-cuda: the tier is DEFERRED — {objects} object(s) of {bytes} byte(s) \
                 served out of {path:?} where they lie while a background thread builds \
                 the page-locked copy; until it lands, a T1 read is a page fault"
            );
            tier.arm_refill(filling, filled);
        }
        Err(why) => eprintln!(
            "engine-cuda: the deferred tier's fill thread would not start ({why}); it \
             will serve out of the artifact for the life of this load"
        ),
    }
}

/// The fill, on its own thread: bind, map, read, verify, page-lock, send.
/// Page-lock runs last since `cudaHostAlloc` holds the memory-manager lock
/// while it runs. A failure sends nothing; the seat keeps serving what it had.
fn refill_from(
    refill: &crate::checkpoint_serving::Landings,
    bytes: usize,
    ordinal: i32,
    out: &std::sync::mpsc::Sender<crate::device::Pinned>,
) {
    let path = &refill.path;
    if let Err(why) = crate::device::ctx::bind_thread(ordinal) {
        eprintln!("engine-cuda: the deferred tier's fill cannot bind device {ordinal} ({why})");
        return;
    }
    // Uninitialized and not yet page-locked; page-lock happens at the end.
    let host = match crate::device::Pinning::uninit(bytes) {
        Ok(host) => host,
        Err(why) => {
            eprintln!(
                "engine-cuda: the deferred tier's fill could not map {bytes} bytes \
                 ({why}); the seat serves out of {path:?} for the life of this load"
            );
            return;
        }
    };
    // SAFETY: `host` maps exactly `bytes`, which `host_layout` tiles, and
    // was made on this thread with no other reader until it is sent.
    match unsafe { crate::checkpoint_serving::read_into(refill, host.host()) } {
        // The lock is taken only over already-verified bytes.
        Ok(()) => match host.lock() {
            Ok(host) => {
                let _ = out.send(host);
            }
            Err(why) => eprintln!(
                "engine-cuda: the deferred tier's fill could not page-lock {bytes} bytes \
                 ({why}); the seat serves out of {path:?} for the life of this load"
            ),
        },
        // A performance failure, not correctness: the seat still serves the verified mapping.
        Err(why) => eprintln!(
            "engine-cuda: the deferred tier's fill could not read {path:?} back ({why}); \
             the seat serves out of the mapping and the file is left alone. {}",
            checkpoint::serving::rebuild(None),
        ),
    }
}

/// The transform arena's backing: RAM when the machine has room, a
/// file-backed map when it does not. See [`Scratch::fitting`].
enum Scratch {
    Ram(Vec<u8>),
    Disk(SpillArena),
}

impl Scratch {
    /// `arena` bytes of scratch, spilled to disk when RAM will not hold it
    /// with a safety share left for the rest of the load. `mapped` is the
    /// mapped-tier bytes the kernel will page in behind the executor, charged
    /// as headroom since no allocation accounts for them.
    fn fitting(arena: usize, mapped: u64) -> Result<Scratch> {
        let need = arena as u64 + mapped + (2 << 30);
        if need <= available_memory() {
            return Ok(Scratch::Ram(vec![0u8; arena]));
        }
        eprintln!(
            "engine-cuda: the load's {arena}-byte transform arena does not fit \
             what is left of this machine's memory beside its {mapped} mapped \
             bytes; spilling the arena to disk"
        );
        SpillArena::new(arena).map(Scratch::Disk)
    }

    fn as_mut(&mut self) -> &mut [u8] {
        match self {
            Scratch::Ram(vec) => vec.as_mut_slice(),
            Scratch::Disk(map) => map.as_mut(),
        }
    }
}

/// The tighter of the kernel's `MemAvailable` and this cgroup's remaining
/// allowance. Never zero: unreadable accounting gets the RAM-arena fallback.
fn available_memory() -> u64 {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let rest = line.strip_prefix("MemAvailable:")?;
                let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
                Some(kb * 1024)
            })
        });
    let cgroup = || -> Option<u64> {
        let max: u64 = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let current: u64 = std::fs::read_to_string("/sys/fs/cgroup/memory.current")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        Some(max.saturating_sub(current))
    }();
    match (meminfo, cgroup) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => u64::MAX,
    }
}

/// A writable file-backed map, sized once and unlinked on drop. `MAP_SHARED`
/// over a temp file makes dirty pages the kernel's problem to reclaim under
/// memory pressure, where an anonymous map of the same size would OOM.
struct SpillArena {
    at: *mut u8,
    len: usize,
}

// SAFETY: `at` is a private MAP_SHARED mapping this struct alone owns; the
// file behind it is unlinked at creation, so no other process can reach it.
unsafe impl Send for SpillArena {}

impl SpillArena {
    fn new(len: usize) -> Result<SpillArena> {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("pie-arena-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|why| Fault::Load(checkpoint::error::Error::Checkpoint(format!(
                "the arena spill file {} does not open: {why}",
                path.display()
            ))))?;
        // Unlinked immediately: a crashed load leaves no file behind.
        let _ = std::fs::remove_file(&path);
        file.set_len(len as u64).map_err(|why| {
            Fault::Load(checkpoint::error::Error::Checkpoint(format!(
                "the arena spill file does not grow to {len} bytes: {why}"
            )))
        })?;
        // SAFETY: a fresh shared mapping over a file this fn just created and
        // sized; length and protections are stated, fd may close after mmap.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len.max(1),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                std::os::fd::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Load(checkpoint::error::Error::Checkpoint(
                "the arena spill file does not map".to_string(),
            )));
        }
        Ok(SpillArena {
            at: at.cast(),
            len,
        })
    }

    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is `len` writable bytes this struct owns.
        unsafe { std::slice::from_raw_parts_mut(self.at, self.len) }
    }
}

impl Drop for SpillArena {
    fn drop(&mut self) {
        // SAFETY: unmapping the mapping this struct created.
        unsafe {
            libc::munmap(self.at.cast(), self.len.max(1));
        }
    }
}

/// One quantized weight's other planes, as rows of [`places`]: the scales
/// always, and — for an affine scheme, whose codes centre on a stored zero
/// point — the biases beside them.
#[derive(Debug, Clone, Copy)]
struct Pairing {
    scales: usize,
    biases: Option<usize>,
}

/// The split-plane pairings this load plan states, by the code plane's own
/// name. Read here, never reconstructed by matching a `.scales` suffix.
/// # Errors
/// [`Fault::Param`] for an attachment naming a plane this trace does not
/// declare, or a scale form this shell has no point for.
fn pairings<'a>(
    landing: &'a LoadPlan,
    index: &BTreeMap<&str, usize>,
) -> Result<BTreeMap<&'a str, Pairing>> {
    // Keyed by the id's own number: `TensorId` is `Hash`+`Eq`, not `Ord`.
    let named: BTreeMap<u32, &str> = landing
        .tensors
        .iter()
        .map(|decl| (decl.id.0, decl.name.as_str()))
        .collect();
    let mut out = BTreeMap::new();
    for attachment in &landing.attachments {
        let Some(name) = named.get(&attachment.tensor.0) else {
            continue;
        };
        // Only a plane the trace declares becomes a weight row.
        if !index.contains_key(name) {
            continue;
        }
        let row = |id: TensorId, what: &'static str| -> Result<usize> {
            named
                .get(&id.0)
                .and_then(|plane| index.get(plane))
                .copied()
                .ok_or_else(|| Fault::Param {
                    name: (*name).to_string(),
                    why: what,
                })
        };
        let biases = match attachment.scale_form {
            ScaleForm::RawE8M0 => None,
            // Affine bank: `code * scale + zero`; zero point is required, not optional.
            ScaleForm::Bf16AffineFactors => Some(row(
                attachment.zero_point_tensor.ok_or_else(|| Fault::Param {
                    name: (*name).to_string(),
                    why: "is an affine bank whose attachment names no zero-point \
                          tensor; `code * scale` alone is the wrong centre",
                })?,
                "is an affine bank whose zero points this plan does not publish as a \
                 param of their own",
            )?),
            ScaleForm::F32Factors => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "carries a scale form no point this shell stamps reads: the \
                          cuda plane's split-plane banks are mxfp4 codes under raw \
                          e8m0 exponents and MLX affine codes under bf16 factor pairs",
                });
            }
        };
        out.insert(
            *name,
            Pairing {
                scales: row(
                    attachment.scale_tensor,
                    "is a quantized weight whose scales this plan does not publish as a \
                     param of their own",
                )?,
                biases,
            },
        );
    }
    Ok(out)
}

impl Weights {
    /// Land `contract` against the checkpoint at `path`. A matching device
    /// table under `cache_dir` is read straight to the device, skipping the
    /// executor; a corrupt artifact is never retried, only reported.
    /// # Errors
    /// [`Fault::Load`] for a checkpoint the contract does not fit,
    /// [`Fault::Param`] for a plan and a contract that do not name the same
    /// tensors, [`Fault::Device`] for the residency itself,
    /// [`Fault::Residency`] for a streamed `Serve` with no artifact it can cut.
    pub fn resident(
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        plan: crate::experts::Plan,
        stream: *mut core::ffi::c_void,
        target: StorageTarget,
    ) -> Result<Weights> {
        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };

        // `target` names this rank's band; a rank of one takes all of
        // `Shard::Cut`'s segments.
        let landing = compile(&metadata, contract, target.clone())?;

        // Name -> row map, shared by the landing sink and the table build.
        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();

        let places = places(trace, &plan)?;
        let total = places.last().map_or(0, |p| p.offset + p.reserved);
        let mut store = Buffer::zeroed(usize::try_from(total).unwrap_or(usize::MAX))?;
        // A streamed bank's plane lands whole in the pinned tier, not the store.
        let serving = crate::checkpoint_serving::Serving::open(path, trace);
        // T2 source: this deployment's own artifact, else a resident load's.
        let deferred = defer_tiers(serving.as_ref(), &plan);
        // Whether the checkpoint could fill the image, asked before it exists.
        // The artifact's images are whole tensors landed for one rank; a
        // rank of a wider group wants its band of each, which only the
        // compiled plan below cuts. So a group lands the cold way.
        let restorable = target.tp_size == 1
            && serving
                .as_ref()
                .is_some_and(|serving| serving.covers(&plan.host_layout()).is_ok());
        let source = match plan.spill_demand() > 0 {
            true => serving.clone().map(crate::experts::Spill::Serving),
            false => None,
        };
        // `None` from `defer_tiers` makes the page-locked image up front.
        let mut experts = match plan.streams() {
            true => {
                let fill = match (deferred, restorable) {
                    (Some(artifact), _) => crate::experts::Fill::Deferred(artifact),
                    (None, true) => crate::experts::Fill::Restored,
                    (None, false) => crate::experts::Fill::Cold,
                };
                Some(crate::experts::Tier::open(plan.clone(), source, fill)?)
            }
            false => None,
        };

        // The checkpoint answers first, ahead of the caches below.
        let from_cache = if restorable {
            match serving.as_ref() {
                None => false,
                Some(serving) => {
                    match restore_from_checkpoint(
                        serving,
                        trace,
                        &plan,
                        &places,
                        &mut store,
                        experts.as_mut(),
                    ) {
                        Ok(()) => true,
                        Err(rotten) => {
                            // The restore did not write either; zero both before landing cold.
                            store.zero_span(0, store.bytes())?;
                            if let Some(tier) = experts.as_mut() {
                                tier.zero_host();
                            }
                            let why = match rotten {
                                Rotten::Bytes(why) => format!(
                                    "the checkpoint's own images do not read back ({why}); \
                                     this boot lands them the cold way. {}",
                                    checkpoint::serving::rebuild(None),
                                ),
                                Rotten::Machine(why) => format!(
                                    "this machine could not read the checkpoint's images \
                                     ({why}); this boot lands them the cold way"
                                ),
                            };
                            return Err(Fault::Residency(why));
                        }
                    }
                }
            }
        } else {
            // Lands cold; a spilled load is refused earlier, in `api.rs`.
            false
        };

        let landed = if from_cache {
            // A matched restore wrote exactly the bytes `places` describes.
            vec![true; places.len()]
        } else {
            let mut sink = Landing {
                store: &mut store,
                experts: experts.as_ref(),
                plan: &plan,
                places: &places,
                index: &index,
                landed: vec![false; places.len()],
            };
            // A streamed load has no arena; `landing` is what a key hashes, not this.
            let landed = if plan.streams() {
                let streaming = compile_streaming(&metadata, contract, target)?;
                Execution::new(&streaming, snapshot)
                    .streaming()
                    .sink(&mut sink)
                    .run()?;
                sink.landed
            } else {
                // Host memory; an arena too large for RAM spills to a file-backed map.
                let bytes = usize::try_from(landing.memory.arena_bytes()).unwrap_or(0);
                let mut scratch = Scratch::fitting(bytes, plan.spill_demand())?;
                let mut backing: &mut [u8] = scratch.as_mut();
                Execution::new(&landing, snapshot)
                    .arena(&mut backing)
                    .sink(&mut sink)
                    .run()?;
                let landed = sink.landed;
                drop(scratch);
                landed
            };

            // The device-table artifact is written elsewhere, under `Intent::Prepare` only.
            landed
        };

        // Outside `from_cache` deliberately: a deferred seat is taken on both roads.
        if let Some(tier) = experts.as_mut().filter(|tier| tier.deferring()) {
            crate::experts::count_deferred();
            arm_refill(tier);
        }

        // Empty for every SKU whose weights are all dense.
        let pairings = pairings(&landing, &index)?;

        let mut table = Vec::with_capacity(places.len());
        for (at, place) in places.iter().enumerate() {
            // A registered plane is reserved and zeroed; `register_adapter` fills it.
            if !landed[at] && trace.params[at].source == ParamSource::Checkpoint {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            // A split-plane bank is two handles under one `Def::Weight`,
            // both bound as `U8`: `rows x width` is the byte rectangle.
            let row = match pairings.get(trace.params[at].name.as_str()) {
                // A streamed group carries its seat: the fixed-address cell the select reads.
                Some(pairing) => WeightRow::Planes {
                    // `U4g64tiled` marks the tiled relabelling.
                    repacked: place.dtype == Dtype::U4g64tiled,
                    codes: packed(experts.as_ref(), &store, &places, at)?,
                    scales: packed(experts.as_ref(), &store, &places, pairing.scales)?,
                    biases: match pairing.biases {
                        Some(biases) => {
                            Some(packed(experts.as_ref(), &store, &places, biases)?)
                        }
                        None => None,
                    },
                    seat: experts
                        .as_ref()
                        .and_then(|tier| tier.group_handles(at))
                        .map_or(GroupSeat::RESIDENT, |seat| GroupSeat {
                            cell: seat.cell,
                            hits: seat.hits,
                        }),
                },
                None => {
                    let handle = Tensor::new(
                        address(experts.as_ref(), &store, place.offset, at)?,
                        place.rows,
                        place.width,
                        place.dtype,
                    );
                    match experts.as_ref().and_then(|tier| tier.handles(at)) {
                        None => WeightRow::Dense(handle),
                        Some(handles) => WeightRow::Streamed {
                            slab: handle,
                            table: handles.table,
                            counts: handles.counts,
                        },
                    }
                }
            };
            table.push(Some(row));
        }
        // Resident slots filled from the pinned copy; the rest at pinned bytes over UVA.
        if let Some(tier) = experts.as_mut() {
            let slabs: Vec<u64> = tier
                .plan()
                .banks()
                .iter()
                .map(|bank| store.at(places[bank.param].offset))
                .collect::<Result<_>>()?;
            // Where the store put every plane, so the ladder can displace a T0 group.
            let store_at: Vec<(usize, u64)> = tier
                .plan()
                .seated()
                .iter()
                .flat_map(|group| group.planes.clone())
                .map(|plane| Ok((plane.param, store.at(places[plane.param].offset)?)))
                .collect::<Result<_>>()?;
            tier.land(&slabs, &store_at, stream)?;
        }
        Ok(Weights {
            store,
            table: WeightTable(table),
            banks: banks(trace, &places),
            experts,
            from_cache,
            rotor: None,
        })
    }

    /// Arms the rotating dense pump: copies a spilled dense plane into a
    /// slot whose address never moves, instead of reading it where it lies.
    /// Returns whether a pump was armed; declining one is still correct.
    /// # Errors
    /// [`Fault::Device`] for a slot, event or stream the runtime refused,
    /// [`Fault::Residency`] for a tier that disagrees with the plan.
    pub fn rotate(
        &mut self,
        trace: &Trace,
        compiled: &model_compiler::CompiledModel,
    ) -> Result<bool> {
        let Some(tier) = self.experts.as_ref() else {
            return Ok(false);
        };
        // A spilled dense plane's shape: one plane, not routed, `experts: 0`.
        let candidates: Vec<(usize, u64)> = tier
            .plan()
            .groups()
            .iter()
            .filter(|group| {
                !group.routed
                    && group.held == crate::experts::Held::Pinned
                    && group.planes.len() == 1
            })
            .map(|group| (group.param, group.bytes))
            .collect();
        if candidates.is_empty() {
            return Ok(false);
        }
        let schedule = model_compiler::prefetch::Schedule::of(trace);
        let rotation = match crate::rotate::Rotation::plan(
            &schedule,
            compiled,
            &candidates,
            crate::rotate::SLOT_CAP,
            crate::rotate::ARENA_CAP,
        ) {
            Ok(rotation) => rotation,
            // A decline is not a fault: every plane is read where it lies.
            Err(_why) => return Ok(false),
        };
        // Page-locked source; the tier owns these bytes for the load's life.
        let mut source: Vec<*const u8> = Vec::with_capacity(rotation.tenants().len());
        for tenant in rotation.tenants() {
            let at = tier.serving_host_of(tenant.param).ok_or_else(|| {
                Fault::Residency(format!(
                    "`{}` was planned to rotate and the pinned tier seats no bytes for it",
                    trace.params[tenant.param].name
                ))
            })?;
            source.push(at);
        }
        let rotor = crate::rotate::Rotor::open(rotation, source)?;
        // The rows now name the slots.
        for tenant in rotor.rotation().tenants() {
            let Some(seat) = rotor.seat(tenant.param) else {
                continue;
            };
            let param = &trace.params[tenant.param];
            let (rows, width) = rectangle(&param.shape);
            self.table.0[tenant.param] = Some(WeightRow::Dense(Tensor::new(
                seat,
                u32::try_from(rows).unwrap_or(u32::MAX),
                u32::try_from(width).unwrap_or(u32::MAX),
                param.dtype,
            )));
        }
        self.rotor = Some(rotor);
        Ok(true)
    }

    /// The rotating dense pump this load armed, or `None`.
    #[must_use]
    pub fn rotor(&self) -> Option<&crate::rotate::Rotor> {
        self.rotor.as_ref()
    }

    /// Does this load rotate dense planes during a fire? A rotating load
    /// takes the eager walk, whatever mode the shell is in.
    #[must_use]
    pub fn rotating(&self) -> bool {
        self.rotor.is_some()
    }

    /// Whether routed experts are served off the host (T1/T2): such a fire
    /// stages its routed experts on the eager path, which a body cannot bake.
    #[must_use]
    pub fn hosts_experts(&self) -> bool {
        self.experts
            .as_ref()
            .is_some_and(|tier| tier.plan().host_image() > 0)
    }

    /// The routed-expert tier this load opened, or `None` for a load whose
    /// banks are resident.
    #[must_use]
    pub fn experts(&self) -> Option<&crate::experts::Tier> {
        self.experts.as_ref()
    }

    /// The same, mutably — what the promotion between two fires is driven
    /// through.
    pub fn experts_mut(&mut self) -> Option<&mut crate::experts::Tier> {
        self.experts.as_mut()
    }

    /// Is the whole weight table on the device? What
    /// [`LoadFacts::weights_resident`](engine::load::LoadFacts) reports.
    #[must_use]
    pub fn all_resident(&self) -> bool {
        self.experts.is_none()
    }

    /// Did this table come off the warm-boot artifact? `true` means the
    /// host-side transform pipeline did not run for this load.
    #[must_use]
    pub fn from_cache(&self) -> bool {
        self.from_cache
    }

    /// The digest of what is actually resident on the device — the bytes,
    /// not the size or source. What a gate compares between a cold load and
    /// a warm one.
    /// # Errors
    /// A device failure reading the store back.
    pub fn digest(&self) -> Result<u64> {
        const CHUNK: usize = 8 << 20;
        let total = self.store.bytes() as u64;
        let mut chunk = vec![0u8; CHUNK.min(self.store.bytes().max(1))];
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut at = 0u64;
        while at < total {
            let want = usize::try_from(total - at).unwrap_or(usize::MAX).min(chunk.len());
            let slice = &mut chunk[..want];
            self.store.read(at, slice)?;
            for byte in slice.iter() {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3);
            }
            at += want as u64;
        }
        Ok(hash)
    }

    /// Write one adapter's planes into the banks: a `cudaMemcpy` per plane
    /// onto an address reserved at load. Re-registering zeroes the slot
    /// first, so a skipped plane can't leave a mix of two adapters' bytes.
    /// # Errors
    /// [`Fault::Adapter`] for a bank this plan does not declare, an id past
    /// the bank's capacity, or a plane whose bytes are not one slot's;
    /// [`Fault::Device`] for the copy.
    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        // Checked whole first: a halfway refusal would leave a bank half-written.
        for plane in planes {
            let bank = self.banks.get(plane.bank).ok_or_else(|| Fault::Adapter {
                bank: plane.bank.to_string(),
                why: "is not a bank this plan declares; a bank is a weight the model \
                      text marked `registered`, and this plan marked none by that name"
                    .to_string(),
            })?;
            if id >= bank.adapters {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} adapters and this registration is id {id}; capacity is \
                         a shape the model text declared, so the fix is the model text \
                         and not a retry",
                        bank.adapters
                    ),
                });
            }
            if plane.bytes.len() as u64 != bank.slot {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} bytes per adapter and this plane carries {}; a plane \
                         is one whole slot, zero-padded by the caller past its own rank",
                        bank.slot,
                        plane.bytes.len()
                    ),
                });
            }
        }
        for plane in planes {
            let bank = self.banks[plane.bank];
            let at = bank.offset + u64::from(id) * bank.slot;
            self.store
                .zero_span(at, usize::try_from(bank.slot).unwrap_or(0))?;
            self.store.write(at, plane.bytes)?;
        }
        Ok(())
    }

    /// The banks this load declared: name, capacity, and bytes per slot.
    /// What a caller sizes its planes against, and what a gate asserts on.
    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.banks
            .iter()
            .map(|(name, bank)| (name.as_str(), bank.adapters, bank.slot))
            .collect()
    }

    /// [`Weights::banks`]'s longer twin: name, capacity, slot bytes, the
    /// slot's rectangle and its element size — what [`crate::blob`] needs to
    /// check the out-major statute against.
    #[must_use]
    pub fn seats(&self) -> Vec<BankSeat> {
        self.banks
            .iter()
            .map(|(name, bank)| BankSeat {
                name: name.clone(),
                adapters: bank.adapters,
                slot: bank.slot,
                rows: bank.rows,
                cols: bank.cols,
                elem: bank.elem,
            })
            .collect()
    }

    /// How many adapters this load can hold resident at once: the smallest
    /// capacity any declared bank states (zero if none), since an adapter
    /// occupies one slot of every bank it fills. Concurrent residency, not a
    /// catalog count.
    #[must_use]
    pub fn adapter_seats(&self) -> u32 {
        self.banks
            .values()
            .map(|bank| bank.adapters)
            .min()
            .unwrap_or(0)
    }

    /// The table a fire resolves `Def::Weight(i)` through.
    #[must_use]
    pub fn table(&self) -> &WeightTable {
        &self.table
    }

    /// Every byte the store holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }
}

/// Where a param's bytes actually are: the device store, or the pinned tier
/// for a plane of a packed group the store does not hold. A handle built off
/// `store.at(offset)` for such a plane would silently name the next param's
/// bytes.
fn address(
    tier: Option<&crate::experts::Tier>,
    store: &Buffer,
    offset: u64,
    param: usize,
) -> Result<u64> {
    match tier.and_then(|tier| tier.offloaded_at(param)) {
        Some(elsewhere) => Ok(elsewhere),
        None => store.at(offset),
    }
}

/// One plane of a split-plane bank, as the mxfp4 select reads it: raw bytes,
/// `U8` dtype, byte rectangle.
fn packed(
    tier: Option<&crate::experts::Tier>,
    store: &Buffer,
    places: &[Place],
    param: usize,
) -> Result<Tensor> {
    let place = places[param];
    // Byte rectangle, not element count, matching the `U8`-bound handle.
    let width = match place.dtype {
        Dtype::Mxfp4 => place.width,
        // Two codes to a byte; `plane_bytes` rounds the total up.
        Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => place.width.div_ceil(2),
        // Four codes to a byte.
        Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => place.width.div_ceil(4),
        Dtype::U8g64 => place.width,
        other => model_compiler::arena::elem_bytes(other)
            .and_then(|element| u32::try_from(element).ok())
            .map(|element| place.width.saturating_mul(element))
            .ok_or_else(|| Fault::Param {
                name: format!("param {param}"),
                why: "is a packed plane in a storage element that has no element size",
            })?,
    };
    Ok(Tensor::new(
        address(tier, store, place.offset, param)?,
        place.rows,
        width,
        Dtype::U8,
    ))
}

/// Where one param's plane sits in the store.
#[derive(Debug, Clone, Copy)]
struct Place {
    offset: u64,
    /// The plane's own bytes — the whole tensor the checkpoint publishes.
    bytes: u64,
    /// What the device store gives it, rounded up to the next handle
    /// alignment. Less than `bytes` for a streamed routed bank, whose slab
    /// seats only `resident` of its experts.
    reserved: u64,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

/// The store's layout, decided before a byte is read: stated ahead rather
/// than accumulated, so an arriving plane is checked against the size its
/// own declaration predicted.
fn places(trace: &Trace, plan: &crate::experts::Plan) -> Result<Vec<Place>> {
    let bytes = plane_bytes(trace)?;
    let mut out = Vec::with_capacity(trace.params.len());
    let mut at = 0u64;
    for (index, param) in trace.params.iter().enumerate() {
        let (rows, width) = rectangle(&param.shape);
        let plane = bytes[index];
        // A streamed bank reserves resident slots only; a packed group reserves nothing here.
        let held = if plan.streamed_whole(index) {
            0
        } else {
            match plan.resident(index) {
                Some(resident) if rows > 0 => plane / rows * u64::from(resident),
                _ => plane,
            }
        };
        out.push(Place {
            offset: at,
            bytes: plane,
            reserved: held.next_multiple_of(ALIGN),
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype: param.dtype,
        });
        at += held.next_multiple_of(ALIGN);
    }
    Ok(out)
}

/// A declared shape, read as `rows x width` — the IR's own rule. Honest
/// rather than load-bearing: `linear.matmul` takes its dimensions from the
/// activation and the result, never from the weight.
fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

/// The sink that puts each finalized tensor where the layout said it goes.
struct Landing<'a> {
    store: &'a mut Buffer,
    /// The pinned tier a streamed bank's plane lands in instead of the store.
    experts: Option<&'a crate::experts::Tier>,
    plan: &'a crate::experts::Plan,
    places: &'a [Place],
    index: &'a BTreeMap<&'a str, usize>,
    landed: Vec<bool>,
}

impl TensorSink for Landing<'_> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoadError> {
        let at = *self.index.get(name).ok_or_else(|| {
            LoadError::Contract(format!(
                "the load contract publishes `{name}`, which this plan does not \
                 name — the two were not written from each other"
            ))
        })?;
        let place = self.places[at];
        if bytes.len() as u64 != place.bytes {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {} — a plane read \
                 at the wrong width is a model that computes",
                bytes.len(),
                place.bytes
            )));
        }
        // A mapped group's bytes are already on disk; count landed and drop.
        if self.plan.mapped(at) {
            self.landed[at] = true;
            return Ok(());
        }
        let streamed = self.plan.resident(at).is_some() || self.plan.pinned(at);
        // A deferred seat has no pinned allocation to land into: it serves
        // T1 out of the artifact where it lies until `arm_refill` fills it.
        if streamed && self.experts.as_ref().is_some_and(|tier| tier.deferred_image().is_some()) {
            self.landed[at] = true;
            return Ok(());
        }
        match self.experts.filter(|_| streamed) {
            Some(tier) => {
                let host_at = tier.host_offset(at).ok_or_else(|| {
                    LoadError::Internal(format!(
                        "`{name}` is a streamed routed bank the tier did not seat"
                    ))
                })?;
                if !tier
                    .host()
                    .write(usize::try_from(host_at).unwrap_or(usize::MAX), bytes)
                {
                    return Err(LoadError::Internal(format!(
                        "`{name}` does not fit the pinned tier at offset {host_at}"
                    )));
                }
            }
            None => self
                .store
                .write(place.offset, bytes)
                .map_err(|fault| LoadError::Internal(fault.to_string()))?,
        }
        self.landed[at] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use model_dsl::Platform;

    use super::*;

    #[test]
    fn the_store_is_laid_out_aligned_disjoint_and_in_plan_order() {
        let trace =
            models::sku("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships the SKU").trace;
        let trace = trace(Platform::Cuda);
        let places = places(&trace, &crate::experts::Plan::default())
            .expect("every param of a bf16 SKU has an element size");

        assert_eq!(places.len(), trace.params.len());
        let mut end = 0u64;
        for (place, param) in places.iter().zip(&trace.params) {
            assert!(place.offset >= end, "`{}` overlaps its predecessor", param.name);
            assert_eq!(place.offset % ALIGN, 0, "`{}` is misaligned", param.name);
            assert!(place.bytes > 0, "`{}` reserves nothing", param.name);
            end = place.offset + place.reserved;
        }

        // The embedding is the SKU's largest plane and its first: 248320
        // rows of 1024 bf16, and the head is tied to it, so it is landed once.
        assert_eq!(places[0].offset, 0);
        assert_eq!(places[0].rows, 248_320);
        assert_eq!(places[0].width, 1024);
        assert_eq!(places[0].bytes, 248_320 * 1024 * 2);
    }

}
