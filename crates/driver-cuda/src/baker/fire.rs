//! One baker fire: the walk, and where every value it touches lives.
//!
//! `baker-smoke/src/smoke.rs:824-1000` is the spec. What changes on the way
//! into the driver is exactly the four things the smoke said it was
//! standing in for, and each is marked `REUSED:` where the driver's own
//! machinery took over:
//!
//! | the smoke's | the driver's |
//! |---|---|
//! | one `cudaMalloc`ed page of KV per layer | the real paged pool, through `FireViews::kv` |
//! | one zeroed recurrent slab pair per layer | the real slot-addressed pools, through `FireViews::recurrent` |
//! | three hand-uploaded runtime planes | the fire's own, through `FireViews::streams` |
//! | its own `DecodePlanCache`, replanned per fire | the driver's, raised per fire by `raise_attn_plans` |
//!
//! What did NOT change is the arena, and that is the honest MVP this work
//! ships with rather than around: see [`Fire::arena`].

use core::ffi::c_void;
use std::collections::BTreeMap;

use kernels::raises::Struct;
use kernels::routine::{Cache, In, Refusal};
use kernels_cuda::jit::Ctx;
use kernels_cuda::views::{KvCache, RecurrentState};
use model_compiler::program::{Call, Dt, Program, Slot};
use model_ir::plan::{Op, Plan, ValueId};

use super::Bank;
use super::geometry::Geometry;
use super::marks::Rect;

/// Everything one fire of the baker lane addresses.
pub(crate) struct Fire<'a> {
    pub plan: &'a Plan,
    pub program: &'a Program,
    /// The plane, on this fire's stream and cuBLAS handle.
    pub ctx: &'a Ctx<'a>,
    /// The raw stream, for the D2D an `InOut` point forces.
    pub stream: *mut c_void,
    /// The base of this fire's activation arena.
    ///
    /// `rows * row_pitch` BYTES, WITH NO REUSE, and the no-reuse half is
    /// `model_compiler::program`'s rather than a choice made here: the walk
    /// assigns every result a fresh 16-byte-aligned offset and never frees
    /// one, so `row_pitch` is the sum of every value the lane states. On
    /// qwen35-d0.8b that is **2.45 MiB per row** — for a stack whose
    /// activations are a few hundred KiB live at any moment.
    ///
    /// How the block is CUT is [`Fire::rect`]'s, and it is value-major
    /// rather than the row-major reading `program.rs` describes. The
    /// argument is there; the short version is that the marks a kernel
    /// takes carry no stride, so value-major is the only cut they can
    /// express for a fire of more than one row.
    ///
    /// THE COST, STATED: a 64-row batch is 157 MiB of arena where liveness
    /// analysis would want single-digit MiB. It is affordable at the batch
    /// sizes this lane is being brought up at and it is not affordable at
    /// serving ones. The fix is `program.rs`'s to make — arena liveness and
    /// `InOut` aliasing are on the baker backlog as one item — and when it
    /// lands this field does not change shape, only size.
    ///
    /// NOT the driver's `Scratch::arena`, deliberately: that one is sized
    /// `lowered.arena_bytes` for the LEGACY lowering firing beside this,
    /// and the two layouts are unrelated. Sharing the buffer would mean two
    /// walks writing into one block at offsets neither computed.
    pub arena: *mut c_void,
    /// Rows in this fire — the fire's, never a literal. See `marks::Rect`.
    pub rows: i32,
    /// Requests in this fire; `qo_indptr`'s row count is this.
    pub requests: i32,
    pub banks: &'a BTreeMap<String, Bank>,
    /// The driver's per-fire view arena: the KV pages, the recurrent slabs
    /// and the runtime planes, all bound to this fire's actual request.
    pub views: &'a crate::bind::views::FireViews,
    pub geom: &'a Geometry,
    /// The fa2 decode schedule this fire was raised on.
    pub decode_plan: *const kernels_cuda::attn::fa2::plan::DecodePlanCache,
}

impl<'a> Fire<'a> {
    /// Where a value lives, chasing merges to the arm that survives.
    pub(crate) fn rect(&self, v: ValueId) -> Result<Rect, Refusal> {
        match &self.program.slots[v as usize] {
            Slot::Arena {
                offset,
                width,
                dtype,
                rows: factor,
            } => Ok(Rect {
                // VALUE-MAJOR, AND THAT IS THE WHOLE MULTI-ROW STORY.
                //
                // `program.rs` describes the arena row-major over FIRE ROWS:
                // "every arena slot's row sits at `row * row_pitch +
                // offset`" (`program.rs:22`, and the carve at :288-296). So
                // a value's rows are `row_pitch` apart. But the marks a
                // kernel is handed carry `{ptr, rows, width}` and NO STRIDE
                // (`kernels/src/routine.rs:493-499`), so every kernel reads
                // row `r` at `ptr[r * width]`. For a one-row fire the two
                // agree and the smoke never met the difference. For rows > 1
                // they do not, and the failure is silent: every address
                // stays inside the arena and every launch succeeds.
                //
                // So this reads the SAME offsets value-major instead:
                // value V owns `[offset_V * rows, offset_{V+1} * rows)` and
                // its rows are `width` apart inside it. That is exactly
                // what the strideless marks mean, and it fits, because the
                // walk packs offsets in value order — the block is
                // `(offset_{V+1} - offset_V) * rows`, which is at least
                // `width * size * rows`, and the last one ends at
                // `cursor * rows <= row_pitch * rows`, the arena's size.
                // `offset` is 16-aligned, so `offset * rows` is too.
                //
                // W10 SETTLED THE OTHER HALF and left this reading alone.
                // The remaining disagreement was not here — it was four
                // `Rect::column` cuts into packed rows at the gdn seam, and
                // the answer was not a stride on the mark but the rule the
                // strideless mark already meant: an executor hands out
                // DENSE rectangles only, and a packed row is cut by a
                // kernel that is told the packing. So `rows > 1` fires.
                //
                // The two descriptions of one arena still disagree on
                // paper, and that is `program.rs`'s to retire: value-major
                // is what its offsets mean when the marks are read
                // honestly, and its own prose still says row-major. Sizing
                // does not change either way (`row_pitch * rows`), so the
                // fix is a paragraph and an arena-liveness pass, not a
                // relayout.
                //
                // SAFETY: `offset * rows` and the `rows * width` that
                // follows it are bounded by `row_pitch * rows`, the arena's
                // length, by the argument above.
                ptr: unsafe {
                    self.arena
                        .cast::<u8>()
                        .add(*offset as usize * self.rows as usize)
                        .cast()
                },
                // The slot's row factor rides on top of the fire's count —
                // a routed value has `top_k` rows per fire row (the walk's
                // `Rows::FireTimes`), and the factor is part of the slot's
                // own pitch contribution, so the value-major offset above
                // is already sized for it.
                rows: match factor {
                    model_compiler::program::Rows::Fire => self.rows,
                    model_compiler::program::Rows::FireTimes(k) => self
                        .rows
                        .checked_mul((*k).cast_signed())
                        .ok_or(Refusal::Wide {
                            what: "a routed rectangle's row count",
                            at: i64::from(self.rows) * i64::from(*k),
                            max: i64::from(i32::MAX),
                        })?,
                },
                width: i32::try_from(*width).map_err(|_| Refusal::Wide {
                    what: "an arena rectangle",
                    at: (*width).cast_signed(),
                    max: i64::from(i32::MAX),
                })?,
                dt: *dtype,
            }),
            Slot::Alias(to) => self.rect(*to),
            Slot::Runtime(name) => self.runtime(name),
            // A REFUSAL AND NOT A PANIC. The smoke could `panic!` here; a
            // driver's next line is somebody else's request, and the
            // `guard` boundary would turn it into a status with no name in
            // it.
            Slot::Absent => Err(Refusal::Unstated {
                what: "a value this lane leaves absent, read by a step that runs",
            }),
        }
    }

    /// A runtime plane, off the driver's own per-fire view arena.
    ///
    /// REUSED: `FireViews::streams` is what the legacy walk binds its
    /// `Arg::Named` runtime operands from, so the baker lane reads the same
    /// device buffers the same fire staged — not a copy of them and not a
    /// second upload. `FireStreams::named` is the sanctioned vocabulary and
    /// answers `None` for a null, which is a refusal rather than a fault.
    ///
    /// The rectangles are stated here because `FireStreams` answers an
    /// address and not a shape, and each one's shape is the coupling the
    /// smoke pinned:
    ///
    /// * `token_ids` / `positions` — one i32 per ROW.
    /// * `qo_indptr` — the request CSR, and its ROWS ARE THE REQUEST COUNT,
    ///   not the buffer's length: the appender reads
    ///   `num_requests = qo_indptr.rows` (`kernels-cuda/src/attn/mod.rs:2415`),
    ///   which is what `bind/mod.rs:2206` puts there from `lowered.arg_rows`.
    /// * `row_valid` — one BYTE per row, declared `In<Tensor<i32>>` and cast
    ///   to `*const u8` inside the routine. The declared element is a
    ///   fiction the DECLARATION carries and the buffer must not.
    fn runtime(&self, name: &str) -> Result<Rect, Refusal> {
        let ptr = self
            .views
            .streams
            .named(name)
            .ok_or(Refusal::Absent {
                what: "a runtime plane this fire does not stage",
            })?;
        let (rows, width, dt) = match name {
            "token_ids" | "positions" => (self.rows, 1, Dt::I32),
            "qo_indptr" => (self.requests, self.requests + 1, Dt::I32),
            "row_valid" => (self.rows, 1, Dt::U8),
            _ => {
                return Err(Refusal::Unstated {
                    what: "the rectangle this runtime plane wears",
                });
            }
        };
        Ok(Rect {
            ptr,
            rows,
            width,
            dt,
        })
    }

    /// A runtime plane by name, for a staging shim that needs one the
    /// statement does not carry as an operand.
    pub(crate) fn rect_of_runtime(&self, name: &str) -> Result<Rect, Refusal> {
        self.runtime(name)
    }

    /// This fire's write origin.
    ///
    /// A SCALAR SMUGGLED THROUGH THE POINTER CHANNEL: the appender reads
    /// `first_token.ptr as i32` (`kernels-cuda/src/attn/mod.rs:2423`) and
    /// the driver answers the same way (`bind/views.rs:93`). Zero is a real
    /// origin, and the only one a fire with no peel split ever has — which
    /// is why `FireStreams::named` answers it rather than treating the null
    /// as absent.
    ///
    /// REUSED: taken off the fire's own streams rather than assumed zero,
    /// so a peel tail's origin arrives correctly if this lane ever runs in
    /// one.
    pub(crate) fn first_token(&self) -> i32 {
        self.views.streams.first_token
    }

    pub(crate) fn input(&self, op: &Op, at: usize) -> Result<Rect, Refusal> {
        self.rect(*op.inputs.get(at).ok_or(Refusal::Unstated {
            what: "an operand this statement does not carry",
        })?)
    }

    pub(crate) fn output(&self, op: &Op, at: usize) -> Result<Rect, Refusal> {
        self.rect(*op.outputs.get(at).ok_or(Refusal::Unstated {
            what: "a result this statement does not state",
        })?)
    }

    pub(crate) fn weight(&self, op: &Op, at: usize) -> Result<&Bank, Refusal> {
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
    /// already in it.
    ///
    /// `model_compiler::program` mints a FRESH rectangle for every result,
    /// including the results of the points whose declaration marks an
    /// operand `InOut` (`norm.residual_add`, `rope.partial`,
    /// `gate.sigmoid_mul`). The kernel writes through the pointer it is
    /// handed, so the executor has to put the operand's bytes in the
    /// result's rectangle before it fires — otherwise the launch mutates
    /// the operand and the result's column stays whatever the arena held.
    /// Aliasing the two instead would be a liveness claim this executor has
    /// no analysis to make, and it is the same claim the arena's no-reuse
    /// rule is waiting on.
    pub(crate) fn inout(&self, from: Rect, to: Rect) -> Result<Rect, Refusal> {
        let bytes = from.bytes();
        if bytes > 0 {
            // SAFETY: both rectangles are live spans of this fire's arena,
            // sized by the same width table, and `bytes` is the smaller.
            let code = unsafe {
                cudarc::runtime::sys::cudaMemcpyAsync(
                    to.ptr,
                    from.ptr.cast_const(),
                    bytes,
                    cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    self.stream.cast(),
                )
            };
            if code != cudarc::runtime::sys::cudaError::cudaSuccess {
                return Err(Refusal::Device {
                    why: "the in-place operand could not be staged",
                });
            }
        }
        Ok(to)
    }

    /// The recurrent pool row a statement names.
    ///
    /// REUSED: the driver's own `RecurrentView` for the statement's layer,
    /// which is slot-addressed — `state_base + slot_ids[r] * stride` — so
    /// this binds the REQUEST's state and not a zeroed slab.
    ///
    /// BY LAYER AND NOT BY NAME, which is the one place the baker's cache
    /// vocabulary and the driver's meet. A text names its rows `conv.{l}`
    /// and `delta.{l}` (two `CacheRow::State`s per gdn layer); the driver
    /// keeps ONE `RecurrentView` per layer with both slabs on it —
    /// `conv_slab`/`conv_stride` for the convolution arms and
    /// `slab`/`slot_stride_elems` for the recurrence arms, which the kernels
    /// read disjointly (`kernels-cuda/src/ssm.rs`). So both names resolve to
    /// the same view and the layer is what tells them apart. `Op::layer`
    /// carries that index already, so nothing parses the suffix — a parse
    /// would be a second spelling of a number the plan states.
    pub(crate) fn recurrent(&self, op: &Op) -> Result<Cache<Struct<RecurrentState>>, Refusal> {
        let layer = self.layer(op)?;
        let view = self
            .views
            .recurrent
            .get(layer)
            .ok_or(Refusal::Absent {
                what: "a recurrent slab for the layer this statement names",
            })?;
        Ok(Cache {
            ptr: core::ptr::from_ref(view),
        })
    }

    /// The KV pages a statement names — the request's ACTUAL pages.
    ///
    /// REUSED: `FireViews::kv[layer]`, built by `bind::views::kv_view` from
    /// this fire's `AttnCtx`. Every field that moves per fire — the page
    /// CSR, the write descriptors, `pages_in_batch` — is the fire's own, so
    /// the appender writes where the scheduler said and the decode reads
    /// what is there.
    ///
    /// A RAISE HAS NO SHAPE -- one object with one lifetime, not a
    /// rectangle (`kernels/src/routine.rs:543-560`).
    pub(crate) fn pages(&self, op: &Op) -> Result<In<Struct<KvCache>>, Refusal> {
        let layer = self.layer(op)?;
        let view = self
            .views
            .kv
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or(Refusal::Absent {
                what: "a kv page table for the layer this statement names",
            })?;
        Ok(In {
            ptr: core::ptr::from_ref(view),
            rows: 0,
            width: 0,
        })
    }

    /// Which layer's pool a cache statement names.
    ///
    /// # The layer is in the NAME, and that is a seam
    ///
    /// The driver's pools are `Vec`s indexed by dense model layer:
    /// `FireViews::kv[l]`, `FireViews::recurrent[l]`. The new DSL spells a
    /// layer by putting it in the cache row's NAME — `kv.{l}`, `conv.{l}`,
    /// `delta.{l}` (`model/src/qwen_3_5/forward.rs:36-42`, and every other
    /// family the same way) — while `Op::layer` exists on the statement and
    /// is left `None` by everything except `gemm.attention_landing`
    /// (`model-dsl/src/kernels.rs:29` is its only `.layer(..)` caller).
    ///
    /// So this parses the suffix, and it does so because there is nothing
    /// else to read — not as a cross-check on a number the plan states
    /// twice. **The fix is the DSL's**: `Stmt::cache` knows the row it is
    /// naming and could set `.layer(l)` beside it, at which point this
    /// function is one field read and the parse is deleted. Until then the
    /// parse is the mechanism and is written down as one.
    ///
    /// `Op::layer` is still PREFERRED where it is set, so the day the DSL
    /// fills it this quietly starts using it and the parse becomes the
    /// fallback it should always have been.
    fn layer(&self, op: &Op) -> Result<usize, Refusal> {
        let name = op.cache.as_deref().ok_or(Refusal::Unstated {
            what: "the cache row this statement names",
        })?;
        if let Some(l) = op.layer {
            return Ok(l as usize);
        }
        // `conv.7` -> 7. The separator is the LAST dot, so a family that
        // ever names a row `gdn.conv.7` still resolves.
        name.rsplit_once('.')
            .and_then(|(_, n)| n.parse::<usize>().ok())
            .ok_or(Refusal::Unstated {
                what: "the layer this cache row's name encodes",
            })
    }

    /// One step: the point shim, the staging table, or a refusal naming the
    /// statement.
    pub(crate) fn step(&self, at: u32, call: &Call) -> Result<(), Refusal> {
        let op = &self.plan.ops[at as usize];
        match call {
            Call::Point(point) => super::points_shim::point(self, point, op),
            Call::Symbol(symbol) => super::staging::symbol(self, symbol, op),
            // Named rather than swallowed: a SKU that states a tier-2
            // statement needs a shim written for it, and the refusal says
            // which one.
            Call::Tier2(_) => Err(Refusal::Absent {
                what: "a tier-2 shim; this driver states none",
            }),
        }
    }
}
