//! One baker fire: the walk, and where every value it touches lives.
//!
//! `baker-smoke`'s `Fire` — its `rect`/`input`/`output`/`weight`/`step`
//! block in `smoke.rs` — is the spec. BY NAME AND NOT BY LINE, because the
//! range this cited (`smoke.rs:824-1000`) had drifted onto `Pools::hold`,
//! which is a different thing entirely: a citation across a crate boundary
//! that a rustfmt run can invalidate is a citation that will be wrong.
//!
//! What changes on the way into the driver is exactly the four things the
//! smoke said it was standing in for, and each is marked `REUSED:` where the
//! driver's own machinery took over:
//!
//! | the smoke's | the driver's |
//! |---|---|
//! | one `cudaMalloc`ed page of KV per layer | the real paged pool, through `FireViews::kv` |
//! | one zeroed recurrent slab pair per layer | the real slot-addressed pools, through `FireViews::recurrent` |
//! | three hand-uploaded runtime planes | the fire's own, through `FireViews::streams` |
//! | its own `DecodePlanCache`, replanned per fire | the driver's, raised per fire by `raise_attn_plans` |
//!
//! What did NOT change is the arena's SHAPE — one block, cut by the walk's
//! offsets — and what did change is how much of it there is: the offsets
//! reuse now, so the block is the lane's busiest instant instead of the sum
//! of everything it ever mints. See [`Fire::arena`].

use core::ffi::c_void;
use std::collections::BTreeMap;

use kernels::plane::{Cache, Refusal};
use kernels::raises::Struct;
use kernels_cuda::jit::Ctx;
use kernels_cuda::views::{KvCache, RecurrentState};
use model_compiler::program::{Call, Dt, Program, Slot};
use model_ir::plan::{Op, Plan, ValueId};

use super::Bank;
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
    /// `rows * row_pitch` BYTES, AND THE OFFSETS REUSE. Values whose lives do
    /// not overlap share bytes — the walk's call, not this executor's
    /// (`model_compiler::program::carve`), and this field did not change
    /// shape when it landed, only size: qwen35-d0.8b went from **2.45 MiB
    /// per row** to **487 KiB**, gemma4-31b from **21.8 MiB** to **1 MiB**,
    /// both sitting exactly on the busiest-instant bound. A 64-row batch is
    /// 30 MiB where it was 1.4 GiB.
    ///
    /// How the block is CUT is [`Fire::rect`]'s, and it is value-major: the
    /// marks a kernel takes carry no stride, so value-major is the only cut
    /// they can express for a fire of more than one row. Reuse rides that
    /// reading unchanged, because scaling every per-row offset and every
    /// per-row size by `rows` is a uniform stretch — it preserves exactly
    /// the disjointness (and the sharing) the walk assigned.
    ///
    /// WHAT REUSE COSTS THIS EXECUTOR: nothing at bind time and one rule at
    /// read time — a rectangle is only its value's between the steps
    /// `program::spans` says it is live. Every read here is from a step that
    /// is running, and the single read past the walk (the `out` seam, in
    /// `fire/launch.rs`) is the case the walk holds open to fire end.
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
                // The marks a kernel is handed carry `{ptr, rows, width}`
                // and NO STRIDE (`kernels/src/routine.rs:493-499`), so every
                // kernel reads row `r` at `ptr[r * width]`. A value's rows
                // must therefore be CONTIGUOUS, which is what this reading
                // gives them: value `V` owns
                // `[offset_V * rows, offset_V * rows + bytes_V * rows)` and
                // its rows are `width` apart inside it.
                //
                // A ROW-MAJOR READING WOULD PUT `row_pitch` BETWEEN THEM and
                // agrees with this one only at `rows == 1` — where the smoke
                // lives, which is why it never met the difference. For
                // rows > 1 they disagree and the failure is silent: every
                // address stays inside the arena and every launch succeeds.
                // So `program.rs` states value-major and this reads it; the
                // two descriptions of one arena no longer disagree on paper.
                //
                // REUSE RIDES THIS UNCHANGED. The walk hands out per-row
                // offsets that SHARE between values never live at once, and
                // multiplying every offset and every size by `rows` is a
                // uniform stretch of the per-row layout — it preserves both
                // the sharing and the disjointness exactly. What the sharing
                // asks of this function is nothing: a rectangle is read only
                // while its value is live, which is what a running step and
                // the walk's held-open `out` seam both are.
                //
                // W10 SETTLED THE OTHER HALF and left this reading alone.
                // The remaining disagreement was not here — it was four
                // `Rect::column` cuts into packed rows at the gdn seam, and
                // the answer was not a stride on the mark but the rule the
                // strideless mark already meant: an executor hands out
                // DENSE rectangles only, and a packed row is cut by a
                // kernel that is told the packing. So `rows > 1` fires.
                //
                // SAFETY: the walk keeps `offset + bytes <= row_pitch` for
                // every slot (`program::clashes` is that invariant's guard),
                // so `offset * rows` and the `rows * factor * width` that
                // follows it are bounded by `row_pitch * rows`, the arena's
                // length. `offset` is 16-aligned, so `offset * rows` is too.
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
    ///
    /// # Three names, because three names is what the floor can mint
    ///
    /// A `row_valid` ARM STOOD BESIDE THEM — one BYTE per row, declared
    /// `In<Tensor<i32>>` and cast to `*const u8` inside the routine — and
    /// this function is reached only from `Slot::Runtime(name)`, one line
    /// away in `rect`. `Slot::Runtime` comes from `ValueDef::Runtime`, whose
    /// only producer in the tree is `model-dsl`'s `Recorder::runtime`, and
    /// its seven call sites spell exactly the three names above. So the arm
    /// was unreachable as a SLOT.
    ///
    /// `"row_valid"` is very much alive as a KEY: `kernels-cuda`'s appending
    /// kernels ask for it through `Ctx::staged::<RowValid>()`, which arrives
    /// at `bind::views::FireViews::raised` and never here. The two doors take
    /// different vocabularies, and this one's is the smaller.
    fn runtime(&self, name: &str) -> Result<Rect, Refusal> {
        let ptr = self.views.streams.named(name).ok_or(Refusal::Absent {
            what: "a runtime plane this fire does not stage",
        })?;
        let (rows, width, dt) = match name {
            "token_ids" | "positions" => (self.rows, 1, Dt::I32),
            "qo_indptr" => (self.requests, self.requests + 1, Dt::I32),
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
    ///
    /// AND THE ARENA'S REUSE DID NOT CHANGE THAT — it is what makes the copy
    /// safe. The walk's spans are inclusive at the step that runs
    /// (`program::Span`), so an operand read at step `s` and a result
    /// written at step `s` are live together and can never be given the same
    /// bytes: `from` and `to` are always disjoint here. Aliasing them
    /// instead would be a claim about the kernel's own indexing — whether it
    /// may read a lane of its input after writing that lane of its output —
    /// which is a fact no plan states and no walk can infer. A text that
    /// WANTS the aliasing already has a way to say so: a merge, which
    /// allocates nothing.
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
    /// carries that index, so nothing parses the suffix — a parse would be
    /// a second spelling of a number the plan states.
    pub(crate) fn recurrent(&self, op: &Op) -> Result<Cache<Struct<RecurrentState>>, Refusal> {
        let layer = self.layer(op)?;
        let view = self.views.recurrent.get(layer).ok_or(Refusal::Absent {
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
    /// rectangle (`kernels/src/routine.rs:543-560`), which is what
    /// `Cache::raised` unwraps it to.
    pub(crate) fn pages(&self, op: &Op) -> Result<Cache<Struct<KvCache>>, Refusal> {
        let layer = self.layer(op)?;
        let view = self
            .views
            .kv
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or(Refusal::Absent {
                what: "a kv page table for the layer this statement names",
            })?;
        Ok(Cache {
            ptr: core::ptr::from_ref(view),
        })
    }

    /// Which layer's pool a cache statement addresses.
    ///
    /// # Two columns, one for each half of the question
    ///
    /// The driver's pools are `Vec`s indexed by dense model layer:
    /// `FireViews::kv[l]`, `FireViews::recurrent[l]`. `Op::cache` says a
    /// pool row is addressed AT ALL — it is the name the text's `caches()`
    /// declared and the statement joined to — and `Op::layer` says where in
    /// the tower the statement stands, which is the index those vectors are
    /// keyed by. Both are read here and neither is derived from the other.
    ///
    /// THIS USED TO PARSE THE NAME's suffix (`conv.7` -> 7), because
    /// `Op::layer` was left `None` by every builder except
    /// `gemm.attention_landing` and there was nothing else to read. The
    /// recorder fills it now — a text's `inputs.layers(..)` loop opens the
    /// scope and every statement inside it is stamped
    /// (`model-dsl/src/record.rs`'s `Recorder::at`) — so the parse is gone
    /// and this is one field read, which is what the note standing here
    /// said the fix would be.
    ///
    /// WHERE THE TWO READINGS DIFFER, gemma is the one place and the column
    /// is the truthful one: its kv-sharing layers name another layer's row
    /// (`kv.{source}`), so the suffix answered the row's OWNER while this
    /// answers the layer that is computing. The driver's own view for a
    /// shared layer already reports the source's pages
    /// (`pools/kv_cache_live.rs::layer_view` resolves through
    /// `kv_source_layer`), so indexing at the statement's layer lands on the
    /// same storage by the pool's own alias table rather than by a second
    /// reading of a string.
    fn layer(&self, op: &Op) -> Result<usize, Refusal> {
        if op.cache.is_none() {
            return Err(Refusal::Unstated {
                what: "the cache row this statement names",
            });
        }
        op.layer.map(|l| l as usize).ok_or(Refusal::Unstated {
            what: "the layer this statement is read at",
        })
    }

    /// One step: the generated dispatch, or a refusal naming the statement.
    pub(crate) fn step(&self, at: u32, call: &Call) -> Result<(), Refusal> {
        let op = &self.plan.ops[at as usize];
        match call {
            // THE GENERATED DISPATCH, AND NO SHIM BESIDE IT. Every arm the
            // hand-written `points_shim` carried is emitted from the point's
            // own slot list into `kernels_cuda::points_dispatch`; what stays
            // on this side is `super::bound::Bound`, which says where THIS
            // executor's rectangles live. `baker-smoke` crossed at W5 and was
            // byte-identical across the move; this crossed at R4b, because a
            // claim body that pulls its staging off `Ctx` cannot be reached
            // by an arm that never had a `Ctx` door.
            //
            // AND THE TIER-2 STATEMENT GOES THROUGH THE SAME DOOR, which is
            // what `.wiki/baker.md` draws ("tier-2, same match"). The only
            // difference is where the point it names was declared — an
            // inherent method on `Ctx` rather than a family trait — and that
            // is a fact about the plane crate, settled before this file is
            // compiled. `Call::Tier2` carries the statement with the `cuda::`
            // gate already stripped, which is the name the plane's own table
            // spells. A SHIM STOOD HERE: "a tier-2 shim; this driver states
            // none", refused at load and at fire, and gemma's fused decode
            // arm was unreachable for as long as it did.
            Call::Point(point) | Call::Tier2(point) => kernels_cuda::points_dispatch::dispatch(
                self.ctx,
                &super::bound::Bound {
                    fire: self,
                    op,
                    point,
                },
            ),
            // AND THE STAGING SHIM IS GONE. `baker::staging` was a file whose
            // whole body was this refusal — five arms at W10, four retired by
            // R4b, and a `Box::leak` of the symbol name into a `&'static str`
            // so the last one could still print it. The arms left the way
            // arms leave: `ssm.gdn_prep`, `ssm.gated_delta` and
            // `layout.embed` became claim bodies that state their own
            // operands, and `attn::write_kv_to_pages` and
            // `attn::dispatch_attention_flashinfer_decode` got a `Ctx` door
            // instead (`bind::views::FireViews`).
            //
            // Two canon symbols still reach this arm across the whole catalog
            // — `hc.collapse` and `norm.res_blend`, both argued in
            // `kernels/src/points.rs`, both waiting on something the floor
            // does not have (a `Vararg` mark; a producer for the head-gate
            // logits no text writes). They are refused BY NAME AND AT LOAD
            // now: `baker::resolve::check` reads this variant and reports the
            // symbol, and `serve::load::report_lane` turns that into a
            // refused `load_model` when it stands in the decode lane. A
            // sixty-line file to say the same thing one fire later, without
            // the name, was the thing check-then-bind exists to delete.
            Call::Symbol(_) => Err(Refusal::Absent {
                what: "a staging shim for a canon symbol: the statement's operands \
                       are not the routine's, and this driver states no shim. \
                       `baker::resolve` names the symbol at load",
            }),
        }
    }
}
