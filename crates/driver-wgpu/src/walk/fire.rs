//! One baker fire: the walk, and where every value it touches lives.
//!
//! # `walk.rs`, where cuda's sibling is `fire.rs`
//!
//! The ONE spelling that diverges from `driver-cuda/src/baker/`, and it
//! diverges because `driver-metal` already has a room called `fire` -- the
//! device-half one that holds a fire's scratch leases, its recorded command
//! buffers and its commit path. That crate's `tests/portable_half.rs` keeps the
//! two halves apart by matching PATH SEGMENTS, so a portable module importing
//! `super::fire` reads as the portable half naming the Apple-only half, which
//! is exactly the class of mistake that audit exists to catch. `driver-wgpu`
//! took the same spelling to keep the two shader planes readable side by side,
//! and it is the one this crate keeps.
//!
//! `driver-cuda/src/baker/fire.rs` is the reference; what differs is the
//! payload and nothing else. Cuda's arena is a `*mut c_void` and its rectangles
//! are pointer arithmetic; metal's is an address and an extent; wgpu's is a
//! region that names its BUFFER, because a `wgpu::BufferBinding` is an object
//! and two offsets and there is no address space to do arithmetic in. Here it
//! is [`Plane::Slice`], and the only two things this file asks of one are
//! [`Plane::span`] and [`Plane::extent`].
//!
//! # The walk runs on any host, and that is the design
//!
//! Nothing in this file names a device type, and now nothing in it names a
//! PLANE either. The steps come off a `model_compiler::program::Program`, the
//! rectangles off its `slots`, the pools off a [`Fires::Pools`] the caller
//! supplies as plain regions, and each step goes through [`Fires::dispatch`]
//! with a `ctx` that is `dyn Encode` on both shader planes. What a device does
//! is behind that `dyn`, so the whole walk -- its order, the points it asks
//! for, the handles it binds -- is checkable with no GPU and no adapter in the
//! process, which is what each driver's `tests/the_walk_is_the_program.rs`
//! does.
//!
//! # The arena is value-major, exactly as cuda's is
//!
//! The marks a kernel is handed carry `{handle, rows, width}` and NO STRIDE, so
//! every kernel reads row `r` at element `r * width` of its binding. A value's
//! rows must therefore be CONTIGUOUS, which is what value-major gives them:
//! value `V` owns `[offset_V * rows, offset_V * rows + bytes_V * rows)` and its
//! rows are `width` apart inside it. A ROW-MAJOR READING would put `row_pitch`
//! between them and agrees with this one only at `rows == 1`.
//!
//! [`Plane::Slice`]: crate::Plane::Slice
//! [`Plane::span`]: crate::Plane::span
//! [`Plane::extent`]: crate::Plane::extent
//! [`Fires::Pools`]: crate::Fires::Pools
//! [`Fires::dispatch`]: crate::Fires::dispatch

use core::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use kernels::plane::{Cache, Refusal};
use model_compiler::program::{Call, Dt, Program, Rows, Slot};
use model_ir::plan::{Op, Plan, ValueId};

use crate::walk::marks::{Bindings, Rect};
use crate::walk::{Bank, Fires, Pages, Plane, Recurrent, Runtime};

/// A device-to-device copy this fire needs before a kernel writes through an
/// in-place rectangle.
///
/// A REQUEST AND NOT A CALL, because a copy needs a command encoder and the
/// walk has none. `model_compiler::program` mints a FRESH rectangle for every
/// result, including the results of points whose declaration marks an operand
/// `InOut` (`norm.residual_add`, `rope.partial`, `gate.sigmoid_mul`); the
/// kernel writes through the handle it is handed, so the operand's bytes have
/// to be in the result's region before it fires.
///
/// WHAT A DEVICE HALF DOES WITH ONE IS ITS OWN, and the three answers are the
/// reason this is a record rather than a call. Cuda issues a
/// `cudaMemcpyAsync` on the spot. Metal stages a host `memcpy`, which it can
/// because every allocation is `StorageModeShared`. wgpu can do neither -- a
/// `wgpu::Buffer` is not host-mapped mid-frame -- so it encodes a
/// `copy_buffer_to_buffer` in walk order. All three read the same `{from, to,
/// bytes, op}`, and [`Blit::op`] is what puts it ahead of the right dispatch.
///
/// THE ARENA'S REUSE IS WHAT MAKES THE COPY SAFE, not what threatens it. The
/// walk's spans are inclusive at the step that runs, so an operand read at step
/// `s` and a result written at step `s` are live together and can never be
/// given the same bytes: `from` and `to` are always disjoint. Aliasing them
/// instead would be a claim about the kernel's own indexing, which is a fact no
/// plan states -- and a text that WANTS the aliasing already says so with a
/// merge, which allocates nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blit<S> {
    /// The operand's bytes.
    pub from: S,
    /// The result's rectangle, which the kernel will write through.
    pub to: S,
    /// How many bytes: the operand's, which is the smaller of the two by
    /// construction.
    pub bytes: u64,
    /// Which statement asked, so a refusal or a trace names it -- and so the
    /// device half can stage the copy ahead of the right dispatch.
    pub op: u32,
}

/// HOW BIG THIS FIRE IS, which is the half of a fire no plan holds.
///
/// Four numbers the SCHEDULER decided -- where the activation arena is, how
/// many rows it batched, how many requests those rows belong to, and how deep
/// the tower is -- grouped because a caller that has one has all four and
/// because [`Fire::over`] takes them together with the two things that are the
/// MODEL's (the plan and the banks). Naming the group is also what keeps the
/// two kinds of argument from being read positionally against each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent<S> {
    /// The base of this fire's activation arena, `rows * row_pitch` bytes.
    pub arena: S,
    /// Rows in this fire.
    pub rows: i32,
    /// Requests those rows belong to.
    pub requests: i32,
    /// Layers in the tower, which is how many pool views to mint.
    pub layers: usize,
}

/// Everything one fire of the baker addresses.
pub struct Fire<'a, P: Plane> {
    /// The traced plan: `plan.ops` is what a `Step` indexes.
    pub plan: &'a Plan,
    /// The compiled lane: the steps, and the slot table they index.
    pub program: &'a Program,
    /// The base of this fire's activation arena.
    ///
    /// `rows * row_pitch` BYTES, AND THE OFFSETS REUSE -- values whose lives do
    /// not overlap share bytes, which is the walk's call and not this
    /// executor's (`model_compiler::program::carve`).
    pub arena: P::Slice,
    /// Rows in this fire -- the fire's, never a literal.
    pub rows: i32,
    /// Requests those rows belong to.
    pub requests: i32,
    /// Where every weight the load put on the device is.
    pub banks: &'a BTreeMap<String, Bank<P::Slice>>,
    /// The six planes a `Slot::Runtime` can name, in [`Runtime::ALL`] order.
    ///
    /// RESOLVED ONCE, IN [`Fire::over`], and the staging they came from is not
    /// kept. A fire's tables do not move while it walks -- they are a lease the
    /// driver took before the fire and returns after it -- so reading them
    /// eagerly is the same answer, and it is what lets a fire outlive nothing.
    pub runtime: [Option<P::Slice>; 6],
    /// What every handle this fire minted stands for.
    ///
    /// SHARED WITH THE ENCODER, which is the whole trick: the walk mints, the
    /// body passes the number through, and the driver's `encode` looks it up.
    /// `RefCell` because `BoundOp`'s accessors take `&self` -- a bound
    /// statement is handed to a generated dispatch by shared reference.
    pub bindings: RefCell<Bindings<P::Slice>>,
    /// The copies an `InOut` point forced, in walk order.
    pub blits: RefCell<Vec<Blit<P::Slice>>>,
    /// WHICH STATEMENT IS RUNNING, set by [`Fire::walk`] before each step.
    ///
    /// THE COORDINATES ARE THE WALK'S AND THE DISPATCH IS THE BODY'S, and this
    /// cell is the seam between them. A claim body states an entrypoint and a
    /// grid; where in the plan that statement stands is a fact it has no way to
    /// know and no business stating. `Ctx` is a `dyn Encode` with no room on it
    /// to be told -- so the encoder reads the fire's own cursor, exactly as it
    /// reads the fire's own bindings, and neither is passed through the body.
    pub cursor: Cell<Cursor>,
    /// One paged view per layer, boxed for a stable address.
    ///
    /// A `Cache` mark crosses as a raised address and the body dereferences it,
    /// so what must hold is that the view outlives the body that reads it.
    /// Boxing is what gives each one an address the vector's growth cannot
    /// move, and it is load-bearing rather than tidy.
    pages: Vec<Option<Box<P::PagesView>>>,
    /// The same, for the recurrent slabs.
    recurrent: Vec<Option<Box<P::RecurrentView>>>,
}

impl<'a, P: Plane> Fire<'a, P> {
    /// Address one fire: mint the pool views, and stand ready to walk.
    ///
    /// THE VIEWS ARE BUILT ONCE, UP FRONT, and cuda's are too
    /// (`FireViews::{kv, recurrent}`). They hold HANDLES on a shader plane, so
    /// building one mints bindings -- and a view minted lazily inside an
    /// accessor would mint a fresh set of handles for every statement that
    /// named the layer, which is a longer binding list for no reason.
    /// THE STAGING IS READ AND NOT HELD, which is what keeps `'a` -- the
    /// lifetime of the plan, the program and the banks -- free of the plane's
    /// own. `pools` is borrowed for this call and nothing longer.
    #[must_use]
    pub fn over<'p>(
        plan: &'a Plan,
        program: &'a Program,
        at: Extent<P::Slice>,
        banks: &'a BTreeMap<String, Bank<P::Slice>>,
        pools: &P::Pools,
    ) -> Self
    where
        P: Fires<'p>,
    {
        let mut bindings = Bindings::new();
        let pages = (0..at.layers)
            .map(|l| P::pages(&mut bindings, pools, l as u32).map(Box::new))
            .collect();
        let recurrent = (0..at.layers)
            .map(|l| P::recurrent(&mut bindings, pools, l as u32).map(Box::new))
            .collect();
        Self {
            plan,
            program,
            arena: at.arena,
            rows: at.rows,
            requests: at.requests,
            banks,
            runtime: Runtime::ALL.map(|which| P::table(pools, which)),
            bindings: RefCell::new(bindings),
            blits: RefCell::new(Vec::new()),
            cursor: Cell::new(Cursor::default()),
            pages,
            recurrent,
        }
    }

    /// Where a value lives, chasing merges to the arm that survives.
    ///
    /// PUBLIC WHERE THE OTHER ACCESSORS ARE NOT, and the asymmetry is a real
    /// caller rather than an oversight. `input`, `output`, `weight` and the
    /// scalar readers below are the bound statement's and nothing outside this
    /// crate has a statement to read them against; a DEVICE HALF asks this one,
    /// because the plan's `out` seam names a value and the driver needs the
    /// rectangle it landed in to read the logits back
    /// (`driver-metal/src/serve/launch.rs`).
    ///
    /// # Errors
    ///
    /// A value the program states no slot for, leaves absent, or whose
    /// rectangle would leave this fire's arena.
    pub fn rect(&self, v: ValueId) -> Result<Rect<P::Slice>, Refusal> {
        match self.program.slots.get(v as usize) {
            Some(Slot::Arena {
                offset,
                width,
                dtype,
                rows: factor,
            }) => {
                // The slot's row factor rides on top of the fire's count -- a
                // routed value has `top_k` rows per fire row -- and the factor
                // is part of the slot's own pitch contribution, so the
                // value-major offset below is already sized for it.
                let rows = match factor {
                    Rows::Fire => self.rows,
                    Rows::FireTimes(k) => {
                        self.rows
                            .checked_mul((*k).cast_signed())
                            .ok_or(Refusal::Wide {
                                what: "a routed rectangle's row count",
                                at: i64::from(self.rows) * i64::from(*k),
                                max: i64::from(i32::MAX),
                            })?
                    }
                };
                let width = i32::try_from(*width).map_err(|_| Refusal::Wide {
                    what: "an arena rectangle",
                    at: (*width).cast_signed(),
                    max: i64::from(i32::MAX),
                })?;
                let bytes =
                    u64::from(rows.unsigned_abs()) * u64::from(width.unsigned_abs()) * dtype.size();
                // VALUE-MAJOR. `program::clashes` keeps
                // `offset + bytes <= row_pitch` for every slot, so
                // `offset * rows` and the extent that follows it are bounded by
                // `row_pitch * rows`, which is the arena's length -- and `span`
                // is what checks that rather than assuming it.
                let slice = P::span(
                    self.arena,
                    *offset * u64::from(self.rows.unsigned_abs()),
                    bytes,
                )
                .ok_or(Refusal::Wide {
                    what: "a rectangle that leaves this fire's arena",
                    at: (*offset).cast_signed(),
                    max: P::extent(self.arena).cast_signed(),
                })?;
                Ok(Rect {
                    slice,
                    rows,
                    width,
                    dt: *dtype,
                })
            }
            Some(Slot::Alias(to)) => self.rect(*to),
            Some(Slot::Runtime(name)) => self.runtime(name),
            // A REFUSAL AND NOT A PANIC: a driver's next line is somebody
            // else's request.
            Some(Slot::Absent) => Err(Refusal::Unstated {
                what: "a value this lane leaves absent, read by a step that runs",
            }),
            None => Err(Refusal::Unstated {
                what: "a value the program states no slot for",
            }),
        }
    }

    /// A runtime plane, off the driver's own per-fire staging.
    ///
    /// The rectangles are stated here because the staging answers a REGION and
    /// not a shape, and each one's shape is the coupling cuda's sibling pinned:
    ///
    /// * `token_ids` / `positions` -- one i32 per ROW.
    /// * `qo_indptr` -- the request CSR, and its ROWS ARE THE REQUEST COUNT.
    /// * `row_valid` -- one BYTE per row, declared `In<Tensor<i32>>` and read
    ///   as bytes inside the kernel. The declared element is a fiction the
    ///   DECLARATION carries and the buffer must not.
    ///
    /// THE SHAPE IS HERE AND THE REGION IS THE PLANE'S, which is the split
    /// [`Runtime`] exists to make: a driver's staging enum is wider than these
    /// six and answers for planes no statement names directly, so what crosses
    /// is the closed set a `Slot::Runtime` can ask for.
    fn runtime(&self, name: &str) -> Result<Rect<P::Slice>, Refusal> {
        let (which, rows, width, dt) = match name {
            "token_ids" => (Runtime::TokenIds, self.rows, 1, Dt::I32),
            "positions" => (Runtime::Positions, self.rows, 1, Dt::I32),
            "request_of_token" => (Runtime::RequestOfToken, self.rows, 1, Dt::I32),
            "qo_indptr" => (Runtime::QoIndptr, self.requests, self.requests + 1, Dt::I32),
            "row_valid" => (Runtime::RowValid, self.rows, 1, Dt::U8),
            "sampling_indices" => (Runtime::SamplingIndices, self.requests, 1, Dt::I32),
            _ => {
                return Err(Refusal::Unstated {
                    what: "the rectangle this runtime plane wears",
                });
            }
        };
        let slice = self.runtime[which.at()].ok_or(Refusal::Absent {
            what: "a runtime plane this fire does not stage",
        })?;
        Ok(Rect {
            slice,
            rows,
            width,
            dt,
        })
    }

    pub(crate) fn input(&self, op: &Op, at: usize) -> Result<Rect<P::Slice>, Refusal> {
        self.rect(*op.inputs.get(at).ok_or(Refusal::Unstated {
            what: "an operand this statement does not carry",
        })?)
    }

    pub(crate) fn output(&self, op: &Op, at: usize) -> Result<Rect<P::Slice>, Refusal> {
        self.rect(*op.outputs.get(at).ok_or(Refusal::Unstated {
            what: "a result this statement does not state",
        })?)
    }

    pub(crate) fn weight(&self, op: &Op, at: usize) -> Result<&Bank<P::Slice>, Refusal> {
        let name = op.weights.get(at).ok_or(Refusal::Unstated {
            what: "a weight this statement does not name",
        })?;
        self.banks.get(name).ok_or(Refusal::Absent {
            what: "a bank the load did not put on the device",
        })
    }

    pub(crate) fn p32(op: &Op, at: usize) -> Result<u32, Refusal> {
        let v = op.params.get(at).ok_or(Refusal::Unstated {
            what: "a scalar this statement does not state",
        })?;
        u32::try_from(*v).map_err(|_| Refusal::Wide {
            what: "a stated scalar",
            at: v.cast_signed(),
            max: i64::from(u32::MAX),
        })
    }

    pub(crate) fn pf32(op: &Op, at: usize) -> Result<f32, Refusal> {
        Ok(f32::from_bits(Self::p32(op, at)?))
    }

    /// The result rectangle of an `InOut` point, with the operand's bytes
    /// scheduled into it. See [`Blit`].
    pub(crate) fn inout(
        &self,
        from: Rect<P::Slice>,
        to: Rect<P::Slice>,
        op: u32,
    ) -> Result<Rect<P::Slice>, Refusal> {
        let bytes = from.bytes();
        if bytes > to.bytes() {
            return Err(Refusal::Wide {
                what: "an in-place operand wider than the result it is staged into",
                at: bytes.cast_signed(),
                max: to.bytes().cast_signed(),
            });
        }
        // **A RECTANGLE IS NOT COPIED ONTO ITSELF.** When the compiler gives an
        // in-place statement's result the slot its operand already holds --
        // which is what `InOut` means, and what `kernels_macros` says in as
        // many words ("an `InOut` result is the rectangle its operand already
        // is") -- the two slices are the same region and the copy is a no-op
        // that still costs a dispatch and the barrier in front of it.
        //
        // MEASURED, on gpt-oss-20b's decode: 240 of 819 dispatches were these,
        // at ~23 us apiece, which is 5.6 ms of a 19 ms step. The guard is what
        // lets `model_compiler::program::alias_in_place` turn that into
        // nothing rather than into a cheaper copy.
        if bytes > 0 && from.slice != to.slice {
            self.blits.borrow_mut().push(Blit {
                from: from.slice,
                to: to.slice,
                bytes,
                op,
            });
        }
        Ok(to)
    }

    /// The recurrent pool row a statement names.
    ///
    /// BY LAYER AND NOT BY NAME, which is the one place the baker's cache
    /// vocabulary and the driver's meet. A text names its rows `conv.{l}` and
    /// `delta.{l}`; the driver keeps ONE view per layer with all three slabs on
    /// it, which the kernels read disjointly. So both names resolve to the same
    /// view and the layer is what tells them apart -- and `Op::layer` carries
    /// that index, so nothing parses a suffix.
    pub(crate) fn recurrent<'c>(&self, op: &Op) -> Result<Cache<Recurrent<'c, P>>, Refusal>
    where
        P: Fires<'c>,
    {
        let layer = Self::layer(op)?;
        let view = self
            .recurrent
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or(Refusal::Absent {
                what: "a recurrent slab for the layer this statement names",
            })?;
        Ok(P::recurrent_cache(view.as_ref()))
    }

    /// The paged view a statement names -- the request's ACTUAL pages.
    pub(crate) fn pages<'c>(&self, op: &Op) -> Result<Cache<Pages<'c, P>>, Refusal>
    where
        P: Fires<'c>,
    {
        let layer = Self::layer(op)?;
        let view = self
            .pages
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or(Refusal::Absent {
                what: "a kv page table for the layer this statement names",
            })?;
        Ok(P::pages_cache(view.as_ref()))
    }

    /// Which layer's pool a cache statement addresses.
    ///
    /// TWO COLUMNS, ONE FOR EACH HALF OF THE QUESTION. `Op::cache` says a pool
    /// row is addressed AT ALL -- the name the text's `caches()` declared and
    /// the statement joined to -- and `Op::layer` says where in the tower the
    /// statement stands, which is the index the driver's per-layer vectors are
    /// keyed by. Both are read here and neither is derived from the other.
    fn layer(op: &Op) -> Result<usize, Refusal> {
        if op.cache.is_none() {
            return Err(Refusal::Unstated {
                what: "a pool row this statement does not join to",
            });
        }
        op.layer.map(|l| l as usize).ok_or(Refusal::Unstated {
            what: "the layer tag a cache statement is read at",
        })
    }

    /// Fire one step, through the door its `Call` names.
    ///
    /// THE GENERATED DISPATCH AND NO SHIM BESIDE IT. Every point a plane claims
    /// is emitted from the point's own slot list into that plane's
    /// `points_dispatch`; what stays on this side is [`crate::bound::Bound`],
    /// which says where THIS executor's rectangles live.
    ///
    /// `Call::Tier2` goes through the same door on cuda, where the plane states
    /// an inherent surface. Neither shader plane states one -- `Ctx` is
    /// `dyn Encode`, so there is no type to write an inherent impl ON -- so
    /// `TIER2` is the empty census the generator wrote, the arm exists, and the
    /// match inside refuses by name. That is the honest answer and not a hole.
    ///
    /// `Call::Symbol` refuses outright, and the SENTENCE is the plane's:
    /// [`Plane::NO_SYMBOL_AT_FIRE`] is what a driver says about its own canon
    /// table, because metal has two rows whose staging no statement carries and
    /// wgpu has no table at all.
    ///
    /// # Errors
    ///
    /// Whatever the dispatch refuses, or a refusal naming the call this
    /// executor has no door for.
    ///
    /// [`Plane::NO_SYMBOL_AT_FIRE`]: crate::Plane::NO_SYMBOL_AT_FIRE
    pub fn step<'c>(&self, ctx: &<P as Fires<'c>>::Ctx, at: u32, call: &Call) -> Result<(), Refusal>
    where
        P: Fires<'c>,
    {
        let op = self.plan.ops.get(at as usize).ok_or(Refusal::Unstated {
            what: "a step naming a statement the plan does not hold",
        })?;
        match call {
            Call::Point(point) | Call::Tier2(point) => P::dispatch(
                ctx,
                &crate::walk::bound::Bound::<'_, '_, 'c, P> {
                    fire: self,
                    op,
                    at,
                    point,
                    fires: core::marker::PhantomData,
                },
            ),
            Call::Symbol(_) => Err(Refusal::Absent {
                what: P::NO_SYMBOL_AT_FIRE,
            }),
        }
    }

    /// Walk the whole lane, in the order the program states it.
    ///
    /// ONE PASS, NO REORDERING. `model_compiler::program` already put the steps
    /// in a total order the arena's liveness was carved against, so a walk that
    /// resequenced them would be reading rectangles outside their spans.
    ///
    /// # Errors
    ///
    /// The first step that refuses, with the statement's index attached.
    pub fn walk<'c>(&self, ctx: &<P as Fires<'c>>::Ctx) -> Result<(), Refused>
    where
        P: Fires<'c>,
    {
        for step in &self.program.steps {
            self.cursor.set(Cursor {
                op: step.op,
                layer: self.layer_of(step.op),
            });
            self.step(ctx, step.op, &step.call).map_err(|why| Refused {
                op: step.op,
                kernel: self
                    .plan
                    .ops
                    .get(step.op as usize)
                    .map_or_else(String::new, |o| o.kernel.clone()),
                why,
            })?;
        }
        Ok(())
    }

    /// The layer a statement stands at, as a dispatch's coordinates want it.
    /// `0` for the three statements that have none -- embed, the final norm,
    /// the head -- which is honest: they stand before and after the tower.
    fn layer_of(&self, at: u32) -> u16 {
        self.plan
            .ops
            .get(at as usize)
            .and_then(|o| o.layer)
            .and_then(|l| u16::try_from(l).ok())
            .unwrap_or(0)
    }
}

/// One step the walk could not fire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused {
    /// Index into `plan.ops` -- the statement that asked.
    pub op: u32,
    /// The statement's point, as the plan spells it.
    pub kernel: String,
    /// What the plane answered.
    pub why: Refusal,
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "op {} `{}`: {}", self.op, self.kernel, self.why)
    }
}

impl std::error::Error for Refused {}

/// Where the running statement stands. See [`Fire::cursor`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    /// Index into `plan.ops`.
    pub op: u32,
    /// The layer the statement is read at; `0` for the three that stand outside
    /// the tower -- embed, the final norm, the head.
    pub layer: u16,
}
