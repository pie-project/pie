use std::cell::RefCell;

#[cfg(target_vendor = "apple")]
use kernels_metal::Tensor;
use kernels_metal::{ArgValue, Encode, Error, Fire};
#[cfg(target_vendor = "apple")]
use model_exec::fire::MaskSpan;
use model_ir::ValueId;

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles, Pipelines, handles::NIL};
use crate::error::Fault;
use crate::experts::Tier;
use crate::run::SlotTable;
use crate::window::{At, Windows};

#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

#[cfg(target_vendor = "apple")]
#[allow(
    dead_code,
    reason = "the router's dispatch no longer cuts; kept as the name of the seam"
)]
const ROUTER_FILE: &str = "linear/moe_route.metal";

#[cfg(target_vendor = "apple")]
#[allow(dead_code, reason = "see ROUTER_FILE")]
const ROUTER_POINTS: [&str; 2] = ["router_topk", "hash_route_gather"];

#[cfg(target_vendor = "apple")]
const HASHER_FILE: &str = "attn/ple.metal";

#[cfg(target_vendor = "apple")]
const HASHER_POINT: &str = "ple_ngram_ids";

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Sink<'a> {
    device: &'a Context,
    frame: Held<'a>,
    pipelines: &'a Pipelines,
    handles: &'a Handles,
    cuts: Option<Cuts<'a>>,
}

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
enum Held<'a> {
    Borrowed(&'a Frame),
    Owned(RefCell<Option<Frame>>),
}

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Cuts<'a> {
    place: &'a At,
    at: &'a [Option<ValueId>],
    ngram: &'a [Option<ValueId>],
    slots: &'a SlotTable,
    windows: &'a Windows,
    arena: RefCell<Buffer>,
    tier: Option<&'a RefCell<Tier>>,
    rows: Option<&'a RefCell<crate::gather::Slab>>,
    seen: std::cell::Cell<(u32, u32)>,
    groups: std::cell::Cell<u32>,
}

impl<'a> Cuts<'a> {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        place: &'a At,
        at: &'a [Option<ValueId>],
        ngram: &'a [Option<ValueId>],
        slots: &'a SlotTable,
        windows: &'a Windows,
        arena: Buffer,
        tier: Option<&'a RefCell<Tier>>,
        rows: Option<&'a RefCell<crate::gather::Slab>>,
    ) -> Cuts<'a> {
        Cuts {
            place,
            at,
            ngram,
            slots,
            windows,
            arena: RefCell::new(arena),
            tier,
            rows,
            seen: std::cell::Cell::new((u32::MAX, u32::MAX)),
            groups: std::cell::Cell::new(1),
        }
    }
}

impl<'a> Sink<'a> {
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles,
    ) -> Sink<'a> {
        Sink {
            device,
            frame: Held::Borrowed(frame),
            pipelines,
            handles,
            cuts: None,
        }
    }

    #[must_use]
    pub fn streaming(
        device: &'a Context,
        frame: Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles,
        cuts: Cuts<'a>,
    ) -> Sink<'a> {
        Sink {
            device,
            frame: Held::Owned(RefCell::new(Some(frame))),
            pipelines,
            handles,
            cuts: Some(cuts),
        }
    }

    #[must_use]
    pub fn into_frame(self) -> Option<Frame> {
        match self.frame {
            Held::Borrowed(_) => None,
            Held::Owned(cell) => cell.into_inner(),
        }
    }

    #[cfg(target_vendor = "apple")]
    fn with_frame<T>(&self, body: impl FnOnce(&Frame) -> T) -> T {
        match &self.frame {
            Held::Borrowed(frame) => body(frame),
            Held::Owned(cell) => body(
                cell.borrow()
                    .as_ref()
                    .expect("a segment is open until its cut closes it"),
            ),
        }
    }

    #[cfg(target_vendor = "apple")]
    fn cut(&self, fire: Fire, cuts: &Cuts<'_>) -> Result<(), Error> {
        let Some(tier) = cuts.tier else {
            return Ok(());
        };
        let region = cuts.place.region.get();
        let Some(routes) = region
            .checked_sub(1)
            .and_then(|router| cuts.at.get(router as usize).copied().flatten())
        else {
            return Ok(());
        };
        let hint = tier
            .borrow()
            .hint_for(routes)
            .and_then(|hint| cuts.slots.0.get(hint.0 as usize).copied().flatten());
        self.across(
            fire,
            cuts,
            routes,
            "a routing vector",
            |rect, span, pass, arena| {
                tier.borrow_mut()
                    .segment(arena, self.handles, routes, rect, hint, span, pass)
            },
        )
    }

    #[cfg(target_vendor = "apple")]
    fn cut_rows(&self, fire: Fire, cuts: &Cuts<'_>) -> Result<(), Error> {
        let Some(rows) = cuts.rows else {
            return Ok(());
        };
        let region = cuts.place.region.get();
        let Some(ids) = cuts.ngram.get(region as usize).copied().flatten() else {
            return Ok(());
        };
        self.across(
            fire,
            cuts,
            ids,
            "an n-gram id vector",
            |rect, span, _pass, arena| {
                rows.borrow_mut()
                    .segment(arena, self.handles, ids, rect, span)
                    .map(|()| 1)
            },
        )
    }

    #[cfg(target_vendor = "apple")]
    fn across(
        &self,
        fire: Fire,
        cuts: &Cuts<'_>,
        vector: ValueId,
        what: &str,
        seat: impl FnOnce(Tensor, MaskSpan, (u32, u32), &mut Buffer) -> crate::error::Result<u32>,
    ) -> Result<(), Error> {
        let Held::Owned(cell) = &self.frame else {
            return Ok(());
        };
        let refuse = |fault: Fault| Sink::refuse(fire, fault);
        let frame = cell
            .borrow_mut()
            .take()
            .expect("a segment is open until its cut closes it");
        let waited = std::time::Instant::now();
        frame.commit().map_err(refuse)?;
        let waited = waited.elapsed();
        if let Some(tier) = cuts.tier {
            tier.borrow_mut().note_wait(waited.as_nanos() as u64);
        }
        if crate::diag::on().cut_trace {
            eprintln!("cut-wait: {:.1} ms", waited.as_secs_f64() * 1e3);
        }

        let region = cuts.place.region.get();
        let rect = cuts
            .slots
            .0
            .get(vector.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                refuse(Fault::Unbound {
                    what: format!("value {}, {what} the carve gave no rectangle", vector.0),
                })
            })?;
        let window = cuts.windows.at(region, cuts.place.run.get());
        let span = window.span;
        let pass = (window.pass, window.passes);
        if crate::diag::on().cut_trace {
            eprintln!(
                "cut: region {region} run {}: rows {}..{} of value {} ({what}; rect {} x {}; pass {} of {})",
                cuts.place.run.get(),
                span.row_offset,
                span.row_offset + span.rows,
                vector.0,
                rect.rows,
                rect.width,
                pass.0,
                pass.1
            );
        }
        let groups = seat(rect, span, pass, &mut cuts.arena.borrow_mut()).map_err(refuse)?;
        if pass.0 == 0 {
            cuts.groups.set(groups);
        }

        *cell.borrow_mut() = Some(self.device.frame().map_err(refuse)?);
        Ok(())
    }

    fn refuse(fire: Fire, fault: Fault) -> Error {
        Error::Backend {
            op: fire.entrypoint,
            detail: fault.to_string(),
        }
    }
}

static KERNEL_PROFILE: std::sync::Mutex<std::collections::BTreeMap<String, (u64, u64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

#[cfg(target_vendor = "apple")]
fn profiling() -> bool {
    crate::diag::on().kernel_profile.on()
}

#[must_use]
pub fn kernel_profile() -> Vec<(String, u64, u64)> {
    let mut rows: Vec<(String, u64, u64)> = KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(name, &(ns, n))| (name.clone(), ns, n))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    rows
}

pub fn reset_kernel_profile() {
    KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

#[cfg(target_vendor = "apple")]
fn profile_key(entrypoint: &str, args: &[ArgValue]) -> String {
    if !crate::diag::on().kernel_profile.shaped() {
        return entrypoint.to_string();
    }
    let scalars: Vec<String> = args
        .iter()
        .filter_map(|arg| match arg {
            ArgValue::I32(v) => Some(v.to_string()),
            ArgValue::U32(v) => Some(v.to_string()),
            _ => None,
        })
        .collect();
    format!("{entrypoint} [{}]", scalars.join(","))
}

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn record_kernel(name: &str, seconds: f64) {
    let ns = (seconds * 1e9).max(0.0) as u64;
    let mut table = KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let row = table.entry(name.to_string()).or_insert((0, 0));
    row.0 += ns;
    row.1 += 1;
}

impl Encode for Sink<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(cuts) = &self.cuts {
                let here = (cuts.place.region.get(), cuts.place.run.get());
                if cuts.seen.get() != here {
                    cuts.seen.set(here);
                    self.cut(fire, cuts)?;
                }
                let window = cuts.windows.at(here.0, here.1);
                if window.passes > 1 {
                    let groups = cuts.groups.get().max(1);
                    let in_tail = cuts.place.tail.get();
                    let last = window.pass + 1 == groups;
                    let empty = window.pass >= groups;
                    if (in_tail && !last) || (!in_tail && empty) {
                        return Ok(());
                    }
                }
            }
            let pipeline = self
                .pipelines
                .at(self.device.device(), fire)
                .map_err(|fault| Sink::refuse(fire, fault))?;
            if profiling()
                && let Held::Borrowed(_) = &self.frame
            {
                let refuse = |fault: Fault| Sink::refuse(fire, fault);
                let own = self.device.frame().map_err(refuse)?;
                {
                    let encoder = own.encoder();
                    encoder.setComputePipelineState(&pipeline);
                    for (at, arg) in args.iter().enumerate() {
                        self.bind(encoder, fire, at, *arg)?;
                    }
                    let lanes = MTLSize {
                        width: fire.lanes[0].max(1) as usize,
                        height: fire.lanes[1].max(1) as usize,
                        depth: fire.lanes[2].max(1) as usize,
                    };
                    let group = if fire.group == [0, 0, 0] {
                        crate::device::ctx::threadgroup(&pipeline, fire.lanes)
                    } else {
                        MTLSize {
                            width: fire.group[0].max(1) as usize,
                            height: fire.group[1].max(1) as usize,
                            depth: fire.group[2].max(1) as usize,
                        }
                    };
                    encoder.dispatchThreads_threadsPerThreadgroup(lanes, group);
                }
                let seconds = own.commit_timed().map_err(refuse)?;
                record_kernel(&profile_key(fire.entrypoint, args), seconds);
                return Ok(());
            }
            self.with_frame(|frame| {
                let encoder = frame.encoder();
                encoder.setComputePipelineState(&pipeline);
                for (at, arg) in args.iter().enumerate() {
                    self.bind(encoder, fire, at, *arg)?;
                }
                let lanes = MTLSize {
                    width: fire.lanes[0].max(1) as usize,
                    height: fire.lanes[1].max(1) as usize,
                    depth: fire.lanes[2].max(1) as usize,
                };
                let group = if fire.group == [0, 0, 0] {
                    crate::device::ctx::threadgroup(&pipeline, fire.lanes)
                } else {
                    MTLSize {
                        width: fire.group[0].max(1) as usize,
                        height: fire.group[1].max(1) as usize,
                        depth: fire.group[2].max(1) as usize,
                    }
                };
                encoder.dispatchThreads_threadsPerThreadgroup(lanes, group);
                Ok(())
            })?;
            if profiling()
                && let Held::Owned(cell) = &self.frame
            {
                let refuse = |fault: Fault| Sink::refuse(fire, fault);
                let frame = cell
                    .borrow_mut()
                    .take()
                    .expect("a segment is open until its cut closes it");
                let seconds = frame.commit_timed().map_err(refuse)?;
                record_kernel(&profile_key(fire.entrypoint, args), seconds);
                *cell.borrow_mut() = Some(self.device.frame().map_err(refuse)?);
            }
            if let Some(cuts) = &self.cuts
                && fire.file == HASHER_FILE
                && fire.entrypoint.starts_with(HASHER_POINT)
            {
                self.cut_rows(fire, cuts)?;
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = args;
            Err(Sink::refuse(fire, Fault::Deviceless))
        }
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(NIL))
    }
}

#[cfg(target_vendor = "apple")]
impl Sink<'_> {
    fn bind(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        fire: Fire,
        at: usize,
        arg: ArgValue,
    ) -> Result<(), Error> {
        match arg {
            ArgValue::Buffer(handle) | ArgValue::BufferMut(handle) => {
                if handle == NIL {
                    // SAFETY: binds nil at an index the shader doesn't
                    // dereference on this arm (the `absent` contract).
                    unsafe { encoder.setBuffer_offset_atIndex(None, 0, at) };
                    return Ok(());
                }
                let binding = self.handles.get(handle).ok_or_else(|| {
                    Sink::refuse(
                        fire,
                        Fault::Unbound {
                            what: format!(
                                "handle {handle} at argument {at}, which this fire minted no row for"
                            ),
                        },
                    )
                })?;
                // SAFETY: the row retains its buffer; its offset was
                // bounds-checked when the row was minted.
                unsafe {
                    encoder.setBuffer_offset_atIndex(
                        Some(&*binding.slab().clone()),
                        usize::try_from(binding.offset()).expect("an offset inside a reservation"),
                        at,
                    );
                }
                Ok(())
            }
            ArgValue::I32(v) => self.scalar(encoder, &v, at),
            ArgValue::U32(v) => self.scalar(encoder, &v, at),
            ArgValue::F32(v) => self.scalar(encoder, &v, at),
            ArgValue::Usize(v) => self.scalar(encoder, &v, at),
        }
    }

    fn scalar<T: Copy>(
        &self,
        encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
        value: &T,
        at: usize,
    ) -> Result<(), Error> {
        // SAFETY: `value` is a live local of the caller's frame and
        // `setBytes:length:` copies out of it before returning.
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(value).cast(),
                size_of::<T>(),
                at,
            );
        }
        Ok(())
    }
}
