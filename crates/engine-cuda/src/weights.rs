#[cfg(feature = "cuda")]
pub mod arena;

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::file::read::parse_metadata;
use checkpoint::file::zt;
use checkpoint::plan::{LoadPlan, StorageTarget, compile, compile_streaming};
use checkpoint::types::{ScaleForm, TensorId};
use kernels_cuda::Tensor;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant::OffsetKind;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::experts::Attachments;
use crate::run::{WeightRow, WeightTable};

pub(crate) const ALIGN: u64 = 256;

#[derive(Debug, Clone, Copy)]
pub struct AdapterPlane<'a> {
    pub bank: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Debug)]
pub struct Weights {
    store: Buffer,
    table: WeightTable,
    banks: BTreeMap<String, Bank>,
    experts: Option<crate::experts::Tier>,
    from_cache: bool,
    rotor: Option<crate::rotate::Rotor>,
    decoded: Vec<Buffer>,
}

fn decodes_at_load(param: &model_ir::Param) -> bool {
    param.dtype == Dtype::U8g64 && param.shape.len() == 2
}

#[must_use]
pub fn decoded_dense_bytes(trace: &Trace) -> u64 {
    trace
        .params
        .iter()
        .filter(|param| decodes_at_load(param))
        .map(|param| {
            let (rows, width) = rectangle(&param.shape);
            rows.saturating_mul(width).saturating_mul(2)
        })
        .sum()
}

#[derive(Debug, Clone, Copy)]
struct Bank {
    offset: u64,
    adapters: u32,
    slot: u64,
    rows: u64,
    cols: u64,
    elem: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankSeat {
    pub name: String,
    pub adapters: u32,
    pub slot: u64,
    pub rows: u64,
    pub cols: u64,
    pub elem: u64,
}

fn banks(trace: &Trace, places: &[Place]) -> BTreeMap<String, Bank> {
    trace
        .params
        .iter()
        .zip(places)
        .filter(|(param, _)| param.source == ParamSource::Registered)
        .map(|(param, place)| {
            let adapters =
                u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(u32::MAX);
            let slot = if adapters == 0 {
                0
            } else {
                place.bytes / u64::from(adapters)
            };
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

pub fn device_demand(trace: &Trace) -> Result<u64> {
    let places = places(trace, &crate::experts::Plan::default())?;
    Ok(places
        .last()
        .map_or(0, |place| place.offset + place.reserved))
}

pub(crate) fn plane_bytes(trace: &Trace) -> Result<Vec<u64>> {
    trace
        .params
        .iter()
        .map(|param| {
            let (rows, width) = rectangle(&param.shape);
            Ok(match param.dtype {
                Dtype::Mxfp4 => rows.saturating_mul(width),
                Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => {
                    rows.saturating_mul(width).div_ceil(2)
                }
                Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => {
                    rows.saturating_mul(width).div_ceil(4)
                }
                Dtype::U8g64 => rows.saturating_mul(width),
                Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k => {
                    rows.saturating_mul(width)
                }
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

#[derive(Debug, Clone)]
pub struct Prospect {
    pub planes: Attachments,
    pub ranking: crate::experts::Ranking,
}

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
    Ok(Prospect { ranking, planes })
}

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

fn restore_from_checkpoint(
    serving: &crate::checkpoint_serving::Serving,
    trace: &Trace,
    plan: &crate::experts::Plan,
    places: &[Place],
    store: &mut Buffer,
    tier: Option<&mut crate::experts::Tier>,
) -> std::result::Result<(), Rotten> {
    let layout = tier
        .as_ref()
        .map(|tier| tier.plan().host_layout())
        .unwrap_or_default();
    let seated = tier
        .as_ref()
        .is_some_and(|tier| tier.deferred_image().is_some());
    let refill = match seated || layout.is_empty() {
        true => None,
        false => Some(serving.refill(&layout).map_err(Rotten::Bytes)?),
    };
    let pinned_params: Vec<u32> = layout
        .iter()
        .map(|(param, _, _, _)| u32::try_from(*param).unwrap_or(u32::MAX))
        .collect();

    let base = store
        .at(0)
        .map_err(|why| Rotten::Machine(format!("{why}")))?;
    let mut transfers = Vec::with_capacity(places.len());
    let mut device_params = Vec::with_capacity(places.len());
    for (param, place) in places.iter().enumerate() {
        if place.reserved == 0
            || trace.params.get(param).map(|p| p.source) != Some(ParamSource::Checkpoint)
        {
            continue;
        }
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

    struct Into(*mut u8);
    // SAFETY: the allocation is the tier's, handed to nobody else, and the
    // thread it moves into is its sole writer for as long as the scope is open.
    unsafe impl Send for Into {}

    let mapped: Vec<u32> = plan
        .mapped_layout()
        .iter()
        .map(|(param, _, _, _)| u32::try_from(*param).unwrap_or(u32::MAX))
        .collect();
    let mut hashed = device_params;
    hashed.extend(mapped);
    let (pumped, pinned) = (transfers.len(), pinned_params.len());
    let into = Into(
        tier.as_ref()
            .map_or(std::ptr::null_mut(), |tier| tier.host().host()),
    );
    let (read, moved, verified) = std::thread::scope(|scope| {
        // SAFETY: `into` is the tier's own uninitialized allocation, which
        // `host_layout` tiles exactly, and no other reader names it yet.
        let reading = scope.spawn(move || match &refill {
            None => serving.verify_planes(&pinned_params),
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
                std::thread::scope(|inner| {
                    let hashing = inner.spawn(|| serving.verify_planes(&hashed));
                    let moved = lanes.pump(&transfers);
                    (moved, hashing.join())
                })
            }
        };
        (reading.join(), moved, verified)
    });

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
        if seated {
            "verified where they lie"
        } else {
            "copied and verified"
        },
    );
    Ok(())
}

fn defer_tiers(
    serving: Option<&crate::checkpoint_serving::Serving>,
    plan: &crate::experts::Plan,
    deferred: bool,
) -> Option<crate::checkpoint_serving::Serving> {
    let serving = serving?;
    if !deferred || plan.host_image() == 0 || !crate::experts::pageable_access() {
        return None;
    }
    if let Err(why) = serving.covers(&plan.host_layout()) {
        eprintln!(
            "engine-cuda: the tier is not deferred — {why}, so this boot builds its \
             page-locked image the eager way; the load is unaffected"
        );
        return None;
    }
    Some(serving.clone())
}

enum Rotten {
    Bytes(String),
    Machine(String),
}

fn arm_refill(tier: &mut crate::experts::Tier) {
    let Some(image) = tier.deferred_image() else {
        return;
    };
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
        Ok(()) => match host.lock() {
            Ok(host) => {
                let _ = out.send(host);
            }
            Err(why) => eprintln!(
                "engine-cuda: the deferred tier's fill could not page-lock {bytes} bytes \
                 ({why}); the seat serves out of {path:?} for the life of this load"
            ),
        },
        Err(why) => eprintln!(
            "engine-cuda: the deferred tier's fill could not read {path:?} back ({why}); \
             the seat serves out of the mapping and the file is left alone. {}",
            checkpoint::serving::rebuild(None),
        ),
    }
}

enum Scratch {
    Ram(Vec<u8>),
    Disk(SpillArena),
}

impl Scratch {
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

struct SpillArena {
    map: memmap2::MmapMut,
    len: usize,
}

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
            .map_err(|why| {
                Fault::Load(checkpoint::error::Error::Checkpoint(format!(
                    "the arena spill file {} does not open: {why}",
                    path.display()
                )))
            })?;
        let _ = std::fs::remove_file(&path);
        file.set_len(len as u64).map_err(|why| {
            Fault::Load(checkpoint::error::Error::Checkpoint(format!(
                "the arena spill file does not grow to {len} bytes: {why}"
            )))
        })?;
        // SAFETY: a fresh shared mapping over a file this fn just created,
        // sized, and unlinked, so no other process can reach it.
        let map = unsafe { memmap2::MmapMut::map_mut(&file) }.map_err(|why| {
            Fault::Load(checkpoint::error::Error::Checkpoint(format!(
                "the arena spill file does not map: {why}"
            )))
        })?;
        Ok(SpillArena { map, len })
    }

    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.map[..self.len]
    }
}

#[derive(Debug, Clone, Copy)]
struct Pairing {
    scales: usize,
    biases: Option<usize>,
}

fn pairings<'a>(
    landing: &'a LoadPlan,
    index: &BTreeMap<&str, usize>,
) -> Result<BTreeMap<&'a str, Pairing>> {
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
    pub fn resident(
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        plan: crate::experts::Plan,
        stream: *mut core::ffi::c_void,
        target: StorageTarget,
        decode_dense: bool,
        deferred_tier: bool,
    ) -> Result<Weights> {
        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };

        let landing = compile(&metadata, contract, target.clone())?;

        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();

        let places = places(trace, &plan)?;
        let total = places.last().map_or(0, |p| p.offset + p.reserved);
        let mut store = Buffer::zeroed(usize::try_from(total).unwrap_or(usize::MAX))?;
        let serving = crate::checkpoint_serving::Serving::open(path, trace);
        let deferred = defer_tiers(serving.as_ref(), &plan, deferred_tier);
        let restorable = target.tp_size == 1
            && serving
                .as_ref()
                .is_some_and(|serving| serving.covers(&plan.host_layout()).is_ok());
        let source = match plan.spill_demand() > 0 {
            true => serving.clone().map(crate::experts::Spill::Serving),
            false => None,
        };
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
            false
        };

        let landed = if from_cache {
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
            let landed = if plan.streams() {
                let streaming = compile_streaming(&metadata, contract, target)?;
                Execution::new(&streaming, snapshot)
                    .streaming()
                    .sink(&mut sink)
                    .run()?;
                sink.landed
            } else {
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

            landed
        };

        if let Some(tier) = experts.as_mut().filter(|tier| tier.deferring()) {
            crate::experts::count_deferred();
            arm_refill(tier);
        }

        let pairings = pairings(&landing, &index)?;

        let mut table = Vec::with_capacity(places.len());
        let mut decoded: Vec<Buffer> = Vec::new();
        for (at, place) in places.iter().enumerate() {
            if !landed[at] && trace.params[at].source == ParamSource::Checkpoint {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            let row = match pairings.get(trace.params[at].name.as_str()) {
                Some(pairing)
                    if decode_dense
                        && decodes_at_load(&trace.params[at])
                        && experts
                            .as_ref()
                            .and_then(|tier| tier.group_handles(at))
                            .is_none() =>
                {
                    let codes = packed(experts.as_ref(), &store, &places, at)?;
                    let scales = packed(experts.as_ref(), &store, &places, pairing.scales)?;
                    let biases = match pairing.biases {
                        Some(biases) => Some(packed(experts.as_ref(), &store, &places, biases)?),
                        None => None,
                    };
                    let (n, k) = (place.rows, place.width);
                    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
                    let plane = Buffer::zeroed(bytes)?;
                    // SAFETY: `stream` is the load's stream, live for the load.
                    let ctx = unsafe { kernels_cuda::Ctx::on(stream.cast()) };
                    let tile = kernels_cuda::linear::quant::decode_into(
                        &ctx,
                        "linear.matmul",
                        codes,
                        scales,
                        OffsetKind::Post,
                        biases,
                        Dtype::Bf16,
                        plane.ptr(),
                        n,
                        k,
                    )?;
                    decoded.push(plane);
                    WeightRow::Dense(tile)
                }
                Some(pairing) => WeightRow::Planes {
                    repacked: place.dtype == Dtype::U4g64tiled,
                    codes: packed(experts.as_ref(), &store, &places, at)?,
                    scales: packed(experts.as_ref(), &store, &places, pairing.scales)?,
                    biases: match pairing.biases {
                        Some(biases) => Some(packed(experts.as_ref(), &store, &places, biases)?),
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
        if let Some(tier) = experts.as_mut() {
            let slabs: Vec<u64> = tier
                .plan()
                .banks()
                .iter()
                .map(|bank| store.at(places[bank.param].offset))
                .collect::<Result<_>>()?;
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
            decoded,
        })
    }

    pub fn rotate(
        &mut self,
        trace: &Trace,
        compiled: &model_compiler::CompiledModel,
    ) -> Result<bool> {
        let Some(tier) = self.experts.as_ref() else {
            return Ok(false);
        };
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
            Err(_why) => return Ok(false),
        };
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

    #[must_use]
    pub fn rotor(&self) -> Option<&crate::rotate::Rotor> {
        self.rotor.as_ref()
    }

    #[must_use]
    pub fn rotating(&self) -> bool {
        self.rotor.is_some()
    }

    #[must_use]
    pub fn hosts_experts(&self) -> bool {
        self.experts
            .as_ref()
            .is_some_and(|tier| tier.plan().host_image() > 0)
    }

    #[must_use]
    pub fn experts(&self) -> Option<&crate::experts::Tier> {
        self.experts.as_ref()
    }

    pub fn experts_mut(&mut self) -> Option<&mut crate::experts::Tier> {
        self.experts.as_mut()
    }

    #[must_use]
    pub fn all_resident(&self) -> bool {
        self.experts.is_none()
    }

    #[must_use]
    pub fn from_cache(&self) -> bool {
        self.from_cache
    }

    pub fn digest(&self) -> Result<u64> {
        const CHUNK: usize = 8 << 20;
        let total = self.store.bytes() as u64;
        let mut chunk = vec![0u8; CHUNK.min(self.store.bytes().max(1))];
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut at = 0u64;
        while at < total {
            let want = usize::try_from(total - at)
                .unwrap_or(usize::MAX)
                .min(chunk.len());
            let slice = &mut chunk[..want];
            self.store.read(at, slice)?;
            for byte in slice.iter() {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3);
            }
            at += want as u64;
        }
        Ok(hash)
    }

    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
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

    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.banks
            .iter()
            .map(|(name, bank)| (name.as_str(), bank.adapters, bank.slot))
            .collect()
    }

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

    #[must_use]
    pub fn adapter_seats(&self) -> u32 {
        self.banks
            .values()
            .map(|bank| bank.adapters)
            .min()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn table(&self) -> &WeightTable {
        &self.table
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64 + self.decoded.iter().map(|b| b.bytes() as u64).sum::<u64>()
    }
}

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

fn packed(
    tier: Option<&crate::experts::Tier>,
    store: &Buffer,
    places: &[Place],
    param: usize,
) -> Result<Tensor> {
    let place = places[param];
    let width = match place.dtype {
        Dtype::Mxfp4 => place.width,
        Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => place.width.div_ceil(2),
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

#[derive(Debug, Clone, Copy)]
struct Place {
    offset: u64,
    bytes: u64,
    reserved: u64,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

fn places(trace: &Trace, plan: &crate::experts::Plan) -> Result<Vec<Place>> {
    let bytes = plane_bytes(trace)?;
    let mut out = Vec::with_capacity(trace.params.len());
    let mut at = 0u64;
    for (index, param) in trace.params.iter().enumerate() {
        let (rows, width) = rectangle(&param.shape);
        let plane = bytes[index];
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

fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

struct Landing<'a> {
    store: &'a mut Buffer,
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
        if self.plan.mapped(at) {
            self.landed[at] = true;
            return Ok(());
        }
        let streamed = self.plan.resident(at).is_some() || self.plan.pinned(at);
        if streamed
            && self
                .experts
                .as_ref()
                .is_some_and(|tier| tier.deferred_image().is_some())
        {
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
        let trace = models::sku("qwen35-d0.8b-bf16-kv-bf16")
            .expect("the catalog ships the SKU")
            .trace;
        let trace = trace(Platform::Cuda);
        let places = places(&trace, &crate::experts::Plan::default())
            .expect("every param of a bf16 SKU has an element size");

        assert_eq!(places.len(), trace.params.len());
        let mut end = 0u64;
        for (place, param) in places.iter().zip(&trace.params) {
            assert!(
                place.offset >= end,
                "`{}` overlaps its predecessor",
                param.name
            );
            assert_eq!(place.offset % ALIGN, 0, "`{}` is misaligned", param.name);
            assert!(place.bytes > 0, "`{}` reserves nothing", param.name);
            end = place.offset + place.reserved;
        }

        assert_eq!(places[0].offset, 0);
        assert_eq!(places[0].rows, 248_320);
        assert_eq!(places[0].width, 1024);
        assert_eq!(places[0].bytes, 248_320 * 1024 * 2);
    }
}
