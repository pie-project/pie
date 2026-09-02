//! The encode sink: the engine side of `kernels_metal::Encode`. A kernel
//! entry hands this module a [`Fire`] (shader path, entrypoint, grid) and a
//! flat list of [`ArgValue`]s; the point is resolved to a compiled pipeline
//! ([`Pipelines`]), each argument is bound at its own index, and one
//! dispatch is encoded into the fire's open compute pass. Encode only,
//! never sync: there is no synchronizing call in this file (the only one
//! left in the shell is [`Pending::wait`](crate::device::Pending)).
//!
//! The argument space is one flat positional table, gaps included: a Metal
//! shader declares every parameter at its own `[[buffer(n)]]`, so a slot a
//! shader variant doesn't use is bound [`absent`](kernels_metal::Encode::absent)
//! rather than omitted (an omission would shift every later index). A
//! scalar is bound by value via `setBytes:length:`, not through a buffer.

use std::cell::RefCell;

use kernels_metal::{ArgValue, Encode, Error, Fire};
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
/// The shader file every routing decision is made in.
const ROUTER_FILE: &str = "linear/moe_route.metal";

#[cfg(target_vendor = "apple")]
/// Entrypoints of [`ROUTER_FILE`] that land a routing vector, by prefix:
/// the four ranked `Linear::MoeTopk*` arms lower to `router_topk…`, and
/// `MoeHashRoute` lands the same pair off a table instead of logits.
///
/// This list is load-bearing and its omission is silent: a router whose
/// point isn't named here fires and lands real expert ids, but the tier
/// never rewrites the vector to seat indices, so a streamed load's matmul
/// silently reads another band's bytes.
///
/// `route_sort` (same file) is deliberately excluded: it runs after the ids
/// have already been rewritten.
const ROUTER_POINTS: [&str; 2] = ["router_topk", "hash_route_gather"];

#[cfg(target_vendor = "apple")]
/// The file the gathered class's cut falls after: the n-gram hasher lands a
/// vector of table rows exactly as a router lands a vector of experts.
const HASHER_FILE: &str = "attn/ple.metal";

#[cfg(target_vendor = "apple")]
/// Entrypoints of [`HASHER_FILE`] that land a row vector, by prefix. Same
/// warning as [`ROUTER_POINTS`]: an unlisted point is never cut, so the
/// gather reads a slab of seats at ids that are not seats.
const HASHER_POINT: &str = "ple_ngram_ids";

/// One fire's encode sink: everything a dispatch needs, borrowed — and, for
/// a streamed load, the command buffer itself.
///
/// Built per fire and dropped with it. On a full-residency load it owns
/// nothing, which is what lets `Encode::fire` take `&self`.
///
/// A streamed load owns its frame, because it ends one mid-walk: a segment
/// cut commits the command buffer, waits, swaps seats, and opens the next
/// one — a borrowed `&Frame` can't be committed or replaced. Interior
/// mutability (`RefCell`) rather than `&mut self`, since the walk holds the
/// sink and the dispatch as two separate borrows.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Sink<'a> {
    device: &'a Context,
    frame: Held<'a>,
    pipelines: &'a Pipelines,
    handles: &'a Handles,
    /// `None` for a full-residency load, and then this file is byte for byte
    /// the sink it was before the tier existed.
    cuts: Option<Cuts<'a>>,
}

/// The command buffer, borrowed or owned — see [`Sink`].
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
enum Held<'a> {
    /// A frame the caller opened and will commit: the full-residency path,
    /// and the shape `device_floor`'s probe uses.
    Borrowed(&'a Frame),
    /// The segment in flight. `None` only between the commit that closed one
    /// and the call that opened the next, which is inside one cut.
    Owned(RefCell<Option<Frame>>),
}

/// Everything one segment cut needs, resolved once per fire: a trace fact
/// (`experts::cuts` finds the routers), a fire fact (which region the walk
/// is inside), and a plan fact (where the carve put the routing vector).
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Cuts<'a> {
    /// Which region the walk is inside, and which run of its window.
    place: &'a At,
    /// Per region: the routing vector the router in it writes, or `None`.
    at: &'a [Option<ValueId>],
    /// Per region: the n-gram id vector the hasher in it writes, or `None`.
    /// The gathered class's twin of [`Cuts::at`], and read the same way.
    ngram: &'a [Option<ValueId>],
    /// This fire's arena rectangles — where the routing vector landed.
    slots: &'a SlotTable,
    /// This fire's windows — which rows of it the region just wrote.
    windows: &'a Windows,
    /// A retain of the arena reservation, which is where a routing vector
    /// lives. Shared storage, so reading it is a `memcpy` and rewriting it is
    /// a `memcpy` — no transfer, no staging buffer, no second copy.
    arena: RefCell<Buffer>,
    /// The tier the swap happens in, or `None` for a load that streams no
    /// band and is only here for the gathered class.
    tier: Option<&'a RefCell<Tier>>,
    /// The row slab the gather seats into, or `None` for a load whose
    /// tables are resident. Both this and `tier` may be `Some`: a capped
    /// load can stream experts and gather its n-gram table independently.
    rows: Option<&'a RefCell<crate::gather::Slab>>,
}

impl<'a> Cuts<'a> {
    /// Bind what a cut resolves through.
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
        }
    }
}

impl<'a> Sink<'a> {
    /// Bind the four things a dispatch resolves through.
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

    /// The same sink over a frame it OWNS, cut into segments at `cuts` — the
    /// streamed load's walk.
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

    /// The last segment's command buffer, for the caller to finish (readout
    /// blit, epilogues, async commit). `None` for a borrowed sink, whose
    /// caller already holds the frame.
    #[must_use]
    pub fn into_frame(self) -> Option<Frame> {
        match self.frame {
            Held::Borrowed(_) => None,
            Held::Owned(cell) => cell.into_inner(),
        }
    }

    /// Run `body` against whichever frame this sink holds.
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

    /// The segment cut (`crate::experts`): close this command buffer, wait
    /// for it, swap the seats the segment ahead will read, rewrite the
    /// routing vector to name them, and open the next command buffer.
    ///
    /// The wait is the whole correctness argument: this shell has no fence
    /// and no second copy of the weight store, so what proves "nothing is
    /// reading seat `s`" is that everything committed before this instant
    /// has completed.
    ///
    /// A refusal here surfaces as a `Backend` [`Error`] via [`Sink::refuse`],
    /// since that's the only thing `kernels_metal::Encode` may answer with.
    #[cfg(target_vendor = "apple")]
    fn cut(&self, fire: Fire, cuts: &Cuts<'_>) -> Result<(), Error> {
        let Some(tier) = cuts.tier else {
            return Ok(());
        };
        let region = cuts.place.region.get();
        let Some(routes) = cuts.at.get(region as usize).copied().flatten() else {
            return Ok(());
        };
        // The prediction the router carries, where the carve put it — read
        // at the same cut, beside the routes.
        let hint = tier
            .borrow()
            .hint_for(routes)
            .and_then(|hint| cuts.slots.0.get(hint.0 as usize).copied().flatten());
        self.across(fire, cuts, routes, "a routing vector", |rect, span, arena| {
            tier.borrow_mut()
                .segment(arena, self.handles, routes, rect, hint, span)
        })
    }

    /// The gathered class's cut: everything [`Sink::cut`] says holds here
    /// too. Only the vector's meaning differs — a router's entry is an
    /// expert (tier answers with a seat), a hasher's entry is a table row
    /// (slab answers with a seat) — the rewrite itself is the same.
    #[cfg(target_vendor = "apple")]
    fn cut_rows(&self, fire: Fire, cuts: &Cuts<'_>) -> Result<(), Error> {
        let Some(rows) = cuts.rows else {
            return Ok(());
        };
        let region = cuts.place.region.get();
        let Some(ids) = cuts.ngram.get(region as usize).copied().flatten() else {
            return Ok(());
        };
        self.across(fire, cuts, ids, "an n-gram id vector", |rect, span, arena| {
            rows.borrow_mut()
                .segment(arena, self.handles, ids, rect, span)
        })
    }

    /// Close this command buffer, wait for it, let `seat` rewrite the
    /// vector, and open the next — the half both cuts share. The wait is
    /// what makes the host's `memcpy` of `vector` legal: the bytes are the
    /// device's until everything committed before this instant has completed.
    #[cfg(target_vendor = "apple")]
    fn across(
        &self,
        fire: Fire,
        cuts: &Cuts<'_>,
        vector: ValueId,
        what: &str,
        seat: impl FnOnce(Tensor, MaskSpan, &mut Buffer) -> crate::error::Result<()>,
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
        if let Some(tier) = cuts.tier {
            tier.borrow_mut()
                .note_wait(waited.elapsed().as_nanos() as u64);
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
        let span = cuts
            .windows
            .at(region, cuts.place.run.get())
            .span;
        seat(rect, span, &mut cuts.arena.borrow_mut()).map_err(refuse)?;

        *cell.borrow_mut() = Some(self.device.frame().map_err(refuse)?);
        Ok(())
    }

    /// A shell fault, restated as the [`Error`] a `kernels-metal` entry may
    /// answer with. The entrypoint names the launch; the fault's sentence
    /// becomes the detail, so nothing is lost but the variant.
    fn refuse(fire: Fire, fault: Fault) -> Error {
        Error::Backend {
            op: fire.entrypoint,
            detail: fault.to_string(),
        }
    }
}

/// **THE KERNEL PROFILE** — `PIE_KERNEL_PROFILE=1`, streamed loads only.
///
/// With it set, every dispatch of an owned (segment) frame is committed in
/// its own command buffer and timed on the device (`GPUEndTime -
/// GPUStartTime`), and the time is summed here by entrypoint. It is a
/// measurement mode, not a serving one: a command buffer per kernel costs
/// submission latency the sums do not include, so what the profile answers is
/// "where does the DEVICE time of a token go", kernel by kernel — the
/// question no tool on a box without Xcode can otherwise answer.
static KERNEL_PROFILE: std::sync::Mutex<std::collections::BTreeMap<String, (u64, u64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

#[cfg(target_vendor = "apple")]
fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_KERNEL_PROFILE").is_some_and(|v| v != "0"))
}

/// Every entrypoint the profile has timed: `(name, device ns, launches)`,
/// most device time first. Empty unless `PIE_KERNEL_PROFILE` is set.
#[must_use]
pub fn kernel_profile() -> Vec<(String, u64, u64)> {
    let mut rows: Vec<(String, u64, u64)> = KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(name, &(ns, n))| (name.clone(), ns, n))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    rows
}

/// Forget every timing the profile holds.
pub fn reset_kernel_profile() {
    KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

#[cfg(target_vendor = "apple")]
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
            let pipeline = self
                .pipelines
                .at(self.device.device(), fire)
                .map_err(|fault| Sink::refuse(fire, fault))?;
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
            // ── The kernel profile: this dispatch alone, committed and timed.
            if profiling() {
                if let Held::Owned(cell) = &self.frame {
                    let refuse = |fault: Fault| Sink::refuse(fire, fault);
                    let frame = cell
                        .borrow_mut()
                        .take()
                        .expect("a segment is open until its cut closes it");
                    let seconds = frame.commit_timed().map_err(refuse)?;
                    record_kernel(fire.entrypoint, seconds);
                    *cell.borrow_mut() = Some(self.device.frame().map_err(refuse)?);
                }
            }
            // Segment boundary: a streamed load ends its command buffer here,
            // after the router decided and before the matmuls read. A
            // full-residency load has no `cuts`.
            if let Some(cuts) = &self.cuts {
                if fire.file == ROUTER_FILE
                    && ROUTER_POINTS
                        .iter()
                        .any(|point| fire.entrypoint.starts_with(point))
                {
                    self.cut(fire, cuts)?;
                } else if fire.file == HASHER_FILE
                    && fire.entrypoint.starts_with(HASHER_POINT)
                {
                    self.cut_rows(fire, cuts)?;
                }
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
    /// One argument at one index.
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
            // `size_t` is 64 bits in MSL, which is what the pool's stride
            // seats are declared as.
            ArgValue::Usize(v) => self.scalar(encoder, &v, at),
        }
    }

    /// A scalar bound by value into the encoder's argument storage.
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
