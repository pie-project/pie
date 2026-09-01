//! The encode sink: the engine side of `kernels_metal::Encode`.
//!
//! A kernel entry hands this module a [`Fire`] — a shader path, an
//! entrypoint, and a grid — and a flat list of [`ArgValue`]s. Three things
//! happen and nothing else: the point is resolved to a compiled pipeline
//! ([`Pipelines`]), each argument is bound at its own index, and one
//! dispatch is encoded into the fire's open compute pass. **Encode only,
//! never sync** — decision #15, and on this plane it is structural rather
//! than a discipline: there is no synchronizing call in this file, and the
//! only one left in the whole shell is
//! [`Pending::wait`](crate::device::Pending), which the settle phase calls
//! when it has run out of A/B seats. `Frame::commit` — the spelling that
//! committed and blocked — belongs to the indirect plane and the native
//! surface now; the fire path takes `Frame::commit_async` and does not wait
//! at all.
//!
//! **THE ARGUMENT SPACE IS ONE FLAT POSITIONAL TABLE, GAPS INCLUDED.** A
//! Metal shader declares every parameter — device pointers and `constant`
//! scalars alike — at its own `[[buffer(n)]]`, and `args[i]` binds at index
//! `i`. That is why `kernels-metal`'s entries push
//! [`absent`](kernels_metal::Encode::absent) into the slots a shader's other
//! variant owns (`attn/kv_write.metal` skips 4 and 6–9) instead of omitting
//! them: an omitted slot would shift every argument after it. The sink
//! honours the same rule — a nil handle binds a nil buffer at its index and
//! the count is unchanged.
//!
//! **A scalar is bound by VALUE, not through a buffer.** `setBytes:length:`
//! copies the four or eight bytes into the encoder's own argument storage,
//! which is what a `const constant int&` parameter reads. Staging scalars
//! through a device buffer would be a second allocation per launch and a
//! second thing to keep alive until the command buffer retires.

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

/// **The shader file every routing decision is made in.**
///
/// The segment cut has to fall between the node that DECIDES and the nodes
/// that READ, and both live inside one region — so the region cursor says
/// WHICH mixture and [`ROUTER_POINTS`] says WHEN.
const ROUTER_FILE: &str = "linear/moe_route.metal";

/// **The entrypoints of [`ROUTER_FILE`] that land a routing vector**, by
/// prefix.
///
/// The four ranked `Linear::MoeTopk*` arms all lower to a point named
/// `router_topk…` (`router_topk_f32w_bfloat16`, `router_topk_sigmoid`,
/// `router_topk_sqrt_softplus`). `Linear::MoeHashRoute` lands the SAME
/// `routes`/`weights` pair off a table instead of off logits, and its point is
/// named for what it does rather than for the ranking it does not do — so it
/// is a second prefix here and not a wider one.
///
/// **THIS LIST IS LOAD-BEARING AND ITS OMISSION IS SILENT.** A router whose
/// point is not named here fires, lands real expert ids, and is never cut: the
/// tier never rewrites that vector from expert id to SEAT index, and the
/// selects behind it then index a slab of `slots` seats by an id in
/// `0..experts`. On a full-residency load the two numbers are equal and
/// nothing is visible; on a streamed one the matmul reads another band's
/// bytes and the logits are quietly wrong. `hash_route_gather` was missing
/// here for exactly as long as dsv4-flash could not plan a streamed load at
/// all, which is why nothing caught it.
///
/// `route_sort` lives in the same file and is deliberately outside every
/// prefix: it is the sorted arm's permutation and runs AFTER the ids have
/// already been rewritten.
const ROUTER_POINTS: [&str; 2] = ["router_topk", "hash_route_gather"];

/// One fire's encode sink: everything a dispatch needs, borrowed — and, for a
/// streamed load, the command buffer itself.
///
/// Built per fire and dropped with it, beside the `Run` it is handed to. On a
/// full-residency load it owns nothing — the device, the pass, the pipeline
/// cache and the handle table all outlive it — which is what lets
/// `Encode::fire` take `&self`.
///
/// **A STREAMED LOAD IS THE ONE THAT OWNS ITS FRAME**, because it ENDS one
/// mid-walk. A segment cut commits the command buffer, waits for it, swaps
/// seats on the host and opens the next one; a borrowed `&Frame` cannot be
/// committed (commit consumes) and cannot be replaced, so the streaming
/// constructor takes the frame by value and hands it back through
/// [`Sink::into_frame`] when the walk is over. Interior mutability rather than
/// `&mut self` for the same reason `Encode::fire` takes `&self`: the trait is
/// the kernel plane's and the walk holds the sink and the dispatch as two
/// separate borrows.
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

/// **Everything one segment cut needs**, resolved once per fire.
///
/// The cut is a TRACE fact (`experts::cuts` finds the routers) read through
/// one FIRE fact (which region the walk is inside, which
/// `crate::window::Cursor` writes into [`At`]) against one PLAN fact (where
/// the carve put the routing vector). Nothing here is a kernel name except
/// [`ROUTER_POINTS`], and nothing here is positional in a launch's argument
/// list.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Cuts<'a> {
    /// Which region the walk is inside, and which run of its window.
    place: &'a At,
    /// Per region: the routing vector the router in it writes, or `None`.
    at: &'a [Option<ValueId>],
    /// This fire's arena rectangles — where the routing vector landed.
    slots: &'a SlotTable,
    /// This fire's windows — which rows of it the region just wrote.
    windows: &'a Windows,
    /// A retain of the arena reservation, which is where a routing vector
    /// lives. Shared storage, so reading it is a `memcpy` and rewriting it is
    /// a `memcpy` — no transfer, no staging buffer, no second copy.
    arena: RefCell<Buffer>,
    /// The tier the swap happens in.
    tier: &'a RefCell<Tier>,
}

impl<'a> Cuts<'a> {
    /// Bind what a cut resolves through.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        place: &'a At,
        at: &'a [Option<ValueId>],
        slots: &'a SlotTable,
        windows: &'a Windows,
        arena: Buffer,
        tier: &'a RefCell<Tier>,
    ) -> Cuts<'a> {
        Cuts {
            place,
            at,
            slots,
            windows,
            arena: RefCell::new(arena),
            tier,
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

    /// **The last segment's command buffer**, for the caller to finish: the
    /// readout blit, the epilogues and the asynchronous commit all ride on it
    /// exactly as they do on a full-residency fire.
    ///
    /// `None` for a borrowed sink, whose caller already holds the frame.
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

    /// **The segment cut** (`crate::experts`): close this command buffer,
    /// WAIT for it, swap the seats the segment ahead will read, rewrite the
    /// routing vector to name them, and open the next command buffer.
    ///
    /// The wait is the whole correctness argument. A wired seat is bytes an
    /// already-committed dispatch may still be reading, and this shell has no
    /// fence and no second copy of the weight store — so what proves "nothing
    /// is reading seat `s`" is that everything committed before this instant
    /// has COMPLETED. It also prices the mechanism: the run-ahead collapses to
    /// one on a streamed load, which `serve`'s header records as the stated
    /// trade rather than a regression.
    ///
    /// **A REFUSAL HERE LEAVES THIS FILE AS A `Backend` ERROR**, because the
    /// only thing a `kernels_metal::Encode` may answer with is
    /// [`Error`] — see [`Sink::refuse`], which makes the same trade for every
    /// device refusal discovered mid-encode. The sentence survives whole in
    /// the detail (a segment's seats and its distinct experts, both named);
    /// the [`Fault::Residency`] VARIANT does not, so a caller sorting on the
    /// variant sees this as a fire refusal rather than a load one. The
    /// refusals that can be asked BEFORE a walk — the budget's, the bake's,
    /// the split window's — are all asked before it, in `experts::Plan::of`,
    /// `experts::cuts` and `Shell::walk_streamed`, precisely so that this is
    /// the rare one.
    #[cfg(target_vendor = "apple")]
    fn cut(&self, fire: Fire, cuts: &Cuts<'_>) -> Result<(), Error> {
        let Held::Owned(cell) = &self.frame else {
            return Ok(());
        };
        let region = cuts.place.region.get();
        let Some(routes) = cuts.at.get(region as usize).copied().flatten() else {
            return Ok(());
        };
        let refuse = |fault: Fault| Sink::refuse(fire, fault);
        let frame = cell
            .borrow_mut()
            .take()
            .expect("a segment is open until its cut closes it");
        frame.commit().map_err(refuse)?;

        let rect = cuts
            .slots
            .0
            .get(routes.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                refuse(Fault::Unbound {
                    what: format!(
                        "value {}, a routing vector the carve gave no rectangle",
                        routes.0
                    ),
                })
            })?;
        let span = cuts
            .windows
            .at(region, cuts.place.run.get())
            .span;
        cuts.tier
            .borrow_mut()
            .segment(
                &mut cuts.arena.borrow_mut(),
                self.handles,
                routes,
                rect,
                span,
            )
            .map_err(refuse)?;

        *cell.borrow_mut() = Some(self.device.frame().map_err(refuse)?);
        Ok(())
    }

    /// A shell fault, restated in the vocabulary a kernel entry's caller
    /// speaks.
    ///
    /// Every `kernels-metal` entry answers [`Error`] and nothing else, so a
    /// device refusal discovered mid-encode has to arrive as one — the shell's
    /// `Dispatch*` impls lift the whole family into the contract's
    /// `KernelError` afterwards (`crate::error::kernel`). The
    /// entrypoint is the op name — it is what a reader needs to find the
    /// launch — and the fault's own sentence is the detail, so nothing is
    /// lost but the variant.
    fn refuse(fire: Fire, fault: Fault) -> Error {
        Error::Backend {
            op: fire.entrypoint,
            detail: fault.to_string(),
        }
    }
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
            // ── **THE SEGMENT BOUNDARY** (`crate::experts`). A streamed load
            //    ends its command buffer here — after the router that decided
            //    and before the matmuls that will read — and nothing else in
            //    this file knows the tier exists. A full-residency load has no
            //    `cuts` and this is one `Option` test per dispatch.
            if let Some(cuts) = &self.cuts
                && fire.file == ROUTER_FILE
                && ROUTER_POINTS
                    .iter()
                    .any(|point| fire.entrypoint.starts_with(point))
            {
                self.cut(fire, cuts)?;
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
                    // SAFETY: binding nil at an index the shader either does
                    // not declare or does not dereference on this arm — the
                    // `absent` contract, and what keeps the indices aligned.
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
                // SAFETY: the row retains its buffer, and its offset was
                // bounds-checked against that buffer when the row was minted.
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
