//! The Vulkan execution shell: what it takes to actually FIRE the modules
//! `kernels-vulkan` compiles.
//!
//! `kernels-vulkan` is a table and 665 SPIR-V modules. It knows what each
//! entrypoint's operands are, what its push block looks like, and which device
//! features it needs — and it deliberately knows nothing about instances,
//! queues, descriptor pools or command buffers. This crate is that half.
//!
//! # Why this is not a port of `driver-metal`
//!
//! It shares that crate's vocabulary — the same [`kernels::LaunchRule`], the
//! same `Dims` field names — and it should, because a disagreement about which
//! rule a row names would be a real defect rather than a backend difference.
//!
//! But the thing a rule ANSWERS is not the same. Metal's encoder takes a thread
//! count and a threadgroup and sizes the group at dispatch time.
//! `vkCmdDispatch` takes only a count of workgroups, and how wide one is was
//! decided when `glslc` ran. So the driver's arithmetic is a division by a
//! number it does not choose, against a divisor that varies per module, and
//! [`geometry`] is that division, written down with the reason each rounding
//! goes the way it does.
//!
//! # What the split is for, which is not what `driver-metal`'s is for
//!
//! `driver-metal` is split by what a COMPILER will accept: no Linux host can
//! build an `objc2` message send, so its portable half exists to be buildable
//! away from a Mac.
//!
//! Vulkan is a loader, not a platform. Every line here compiles on every host
//! in the tree. The `native` feature gates what needs a GPU to be PRESENT, so
//! the portable half is defined by what can be PROVED without one — and that is
//! a much better deal than the Metal side got, because the device half is
//! testable on the same machine this crate is written on, against a validation
//! layer that turns a silent misuse into a failed test.
//!
//! # What is here, and the order it was built in
//!
//! [`geometry`] was deliberately first. Every kernel in this tree that was
//! wrong after the Vulkan port was wrong in its LAUNCH SHAPE and not in its
//! arithmetic, because an undershot Vulkan grid writes nothing, leaves the
//! buffer's birth zeros in the gap, and returns success from every call
//! involved. Getting the division right, and being able to check it against
//! each module's own declared workgroup, is the part of a Vulkan shell that
//! carries the defects.
//!
//! [`spirv`] reads a module back: its bindings, its push offsets, its declared
//! workgroup. Every claim this crate makes about a module is measured from the
//! module rather than assumed from the row that names it, and where the two
//! are computed separately they are checked against each other.
//!
//! [`lowering`] and [`dispatch`] turn one of a plan's rectangles into one
//! `vkCmdDispatch`: which buffers, at which offsets, with which scalars, over
//! which grid. [`binding`] is where a row's operand ORDER meets a module's
//! binding order, which are not the same order and were not the same order for
//! 2898 of the 3992 rectangles three real texts state.
//!
//! [`device`] is the Vulkan itself, and the only place in the crate with
//! `unsafe` in it. [`Device::run`](device::Device::run) submits one dispatch
//! and waits; [`run_all`](device::Device::run_all) records a whole plan into
//! one command buffer with a barrier between each pair, and the two are checked
//! to agree over a real plan.
//!
//! [`serve`] is a whole fire: plan every rectangle, allocate every scalar
//! block, build every pipeline, record, submit once, wait. Three passes and
//! not one loop, and the module says why each boundary is where it is. It also
//! holds the last mile, [`serve::logits`]: a fire's distributions are a range
//! of its own arena, and reading them needs an element width the lowering
//! states and a driver must not assume.
//!
//! [`names`] is the one thing that stays per-checkpoint: a plan binds
//! `layer.0.down.zeros` and a loader publishes
//! `layers.0.mlp.down_proj.biases`. Measured against a real compiled load
//! plan, 704 of 704 names disagreed; through this table, none do. It is a
//! table and not a decision -- removing it does not change which kernels fire,
//! only whether they find their operands.
//!
//! [`turns`] is one fire after another over the same cache: grow, frame,
//! lower, stage, fire, read. Everything below it is per-fire, and the things
//! that can only be wrong ACROSS fires -- a conversation's pages, its
//! positions, and a pipeline cache that must stop growing -- live there
//! because there is nowhere else they could.
//!
//! [`pages`] is the book that outlives a fire: which conversation owns which
//! page of the cache. `Frame::of` refuses two requests in one fire naming the
//! same page, but a conversation spans thousands of fires, and handing page
//! numbers out by hand -- which every test here did before it existed -- gives
//! two users each other's history with nothing to notice.
//!
//! [`resources`] and [`rope`] are the memory and the numbers a plan never
//! mentions, because they belong to a DEPLOYMENT rather than to a model: the
//! paged KV cache, the tables a fire assembles, and the rotary ladder a
//! rescaling config asks for. A text that stated any of them would be right for
//! one server and quietly wrong for the next.
//!
//! [`shell`] is one assembled server: it owns the device, the two plans, the
//! cache, the book and the weights, and takes turns. Everything in it existed
//! already and none of it was a server -- until this, every caller assembled
//! thirty lines of pieces that all had to agree, and nothing checked that any
//! pair did. It does not derive the model, because `crates/model` is a
//! dev-dependency here on purpose: a driver executes a text somebody else
//! authored. It CHECKS the pieces instead, which is stronger than deriving
//! -- deriving assumes one set of facts went in and cannot notice when two
//! did. Owning a device also found a leak nothing else could: every caller
//! shared one static device that outlived the process, so nothing had ever
//! destroyed a device with buffers still on it.
//!
//! [`facts`] is the first line the engine's seam reads: what this driver says
//! about the device it opened. `driver-metal` states its as constants and is
//! entitled to; the same Vulkan binary runs on a discrete card, an integrated
//! one and a software implementation, and three of the fields differ. So they
//! are measured, and each is held against the thing HERE that would break if
//! the engine believed a wrong one -- the alignment against the sub-range bind
//! that enforces it, the page size against a pool built at it, the memory
//! question against the heaps rather than against the device kind that
//! answered it.
//!
//! Forking a conversation -- the engine's `copy_kv` -- is deliberately split
//! across three modules and assembled in one. [`pages::Book::fork`] seats the
//! destination on fresh pages and RETURNS the moves rather than performing
//! them; [`resources::Pool::copy_page`] performs one, over both cache sides
//! and every layer; [`device::Device::copy_within`] is the `memmove` under
//! it, on the host because every buffer here is host-coherent. Only
//! [`shell::Shell::fork`] has both halves, which is the point: a caller who
//! could take the seat without copying the bytes would have a conversation
//! attending over whatever those pages last held -- zeros, which are finite
//! and plausible and wrong. Five mutations were tried against the whole-
//! distribution test, including aliasing the source's pages instead of
//! copying them and copying the keys but not the values; all five are
//! refused.
//!
//! The ENGINE's shape for the same thing is different and is served
//! separately: [`resources::Pool::copy_plan`] takes a `KvCopyPlan` -- a list
//! of whole-page moves and a list of single-row cells, addressed by physical
//! page, because the engine's prefix cache has no conversation id to give.
//! Both shapes end at `Pool::copy_rows`. The plan is walked twice, once for
//! refusals and once for work, and the reason is in `driver-metal`'s port
//! too: the C++ applies the page moves first and notices a bad cell
//! afterwards, leaving a cache half somebody else's with no way back.
//!
//! Two findings came out of building it. `PIE_MEMORY_DOMAIN_VULKAN_DEVICE`
//! did not exist -- the ABI named CUDA, ROCm, two Metal domains and pinned
//! host memory, so no plan could be addressed to a Vulkan pool at all; it is
//! added, and the ABI version with it. And `engine::scheduler` states
//! `PIE_MEMORY_DOMAIN_CUDA_DEVICE` on every KV copy plan it builds,
//! regardless of which backend the plan is bound for, which `driver-metal`
//! would refuse as a foreign domain exactly as this does. That is the
//! engine's to fix when the Vulkan seam is built, and it is written down
//! here rather than worked around by accepting any domain.
//!
//! Resizing the cache is the third of the engine's pool verbs, and it is
//! where this driver differs from `driver-metal` most sharply.
//! `driver-metal`'s pool is SPARSE: it commits and releases pages without
//! moving an address, because Metal binds its heap once. Vulkan has sparse
//! binding too and [`resources::Pool::resize`] does not use it --
//! `sparseBinding` is an optional feature, and the premise of
//! [`device::Tier`] is running where the optional features are absent. It
//! reallocates instead, and that is only sound because **every descriptor in
//! this driver is written during the step that uses it**, so no address
//! survives a step for a resize to invalidate. A change that cached
//! descriptor sets across steps must change this too.
//!
//! Writing the test for it found a defect the verb itself did not have:
//! `Shell` kept its own `Shape`, which was stale the instant the pool could
//! be resized -- it reported the old page count while the pool held the new
//! one, so a caller sizing a frame from it would have addressed pages that
//! were no longer there. It asks the pool now.
//!
//! # The one number this crate checks against another number
//!
//! Almost every check here is structural: that a rectangle records, that two
//! recordings of the same plan agree, that a refusal names the right thing.
//! Two tests compare a norm against a host reference, and until recently
//! nothing compared a MATMUL against anything -- a projection that transposed
//! its operands would have passed the whole suite.
//!
//! `the_tiled_gemm_answers_the_way_the_vector_kernel_does` is that check, and
//! it needs no host model: the same text traced at the two fire classes states
//! two different matmul kernels for the same math, so each is the other's
//! reference. Fired at sixteen rows with varying weights, they agree to zero
//! -- bit for bit on all 16 x 151936 logits, for a dense text and for a
//! mixture of experts.
//!
//! It does not cover the ROUTED GEMM, and that is worth stating because the
//! mixture was added expecting it to. `kernels-vulkan` has
//! `affine_qmm_t_routed` and `mxfp4_qmm_t_routed_bias`; no plan this crate can
//! lower states either at any row count, so the expert matmuls stay
//! `affine_qmv_routed` and have no second implementation to be checked
//! against.
//!
//! Two things had to be got right for that to mean anything, and both were
//! found by the comparison failing. The weight blocks cannot be zeros, or an
//! affine dequantisation is a constant. And they cannot be ONE byte repeated
//! either: the scales and the norms are bfloat16, and `0xA3` repeated is a
//! scale of `-2^-56`, which underflows every product to `-0`. The `embed`
//! block also has to be its real size, because it is the tied head and this
//! card returns zero for a read past a buffer.
//!
//! # A plan does not always have one answer
//!
//! `tests/device.rs` fires every text it knows, in both fire classes, twice:
//! once batched into a single command buffer with barriers, and once a
//! submission per dispatch. They are expected to agree byte for byte, and for
//! every dense text and every decode they do.
//!
//! Both mixtures of experts, at a 64-row prefill, do not -- and neither do two
//! runs of the ONE-AT-A-TIME reference against each other. `route_sort` builds
//! its permutation with workgroup-scoped atomics, so when two rows want the
//! same expert the order is whichever lane won, and the gather then writes the
//! same rows to different offsets. A decode is one row and has nothing to tie,
//! which is why nothing here saw it until prefills were fired.
//!
//! So that test runs the reference a second time before it accuses the
//! batching, and only claims byte equality of a plan that has proved itself
//! deterministic. A driver cannot fix this and should not paper over it: it is
//! a property of the kernel, and any caller comparing two runs of a routed
//! prefill needs to know it.
//!
//! **It does not reach the answer.** That was the question worth asking and it
//! is checked rather than argued: two runs of the same routed prefill leave
//! arenas that differ and distributions that do not. The combine reads the
//! permutation back through its inverse, so the shuffle cancels. A varying
//! arena is a curiosity; a varying distribution would be a model that answers
//! the same prompt differently on Tuesday, and this crate now says which one
//! it is.
//!
//! # What every module here has been mutated against
//!
//! Every module in this crate has been swept once: a check is deleted, or a
//! value replaced, or two values swapped, and the suite is run. A mutation
//! that survives is a claim nothing was reading.
//!
//! It found more than it should have. `binding` handed the shader two cache
//! strides nothing read back. Three head-shape overrides fired six hundred
//! times and were checked zero. Five `.max(1)` clamps guarded an input
//! nothing sent -- one was dead, one was hiding a disagreement with the
//! shader, one let a rowless fire dispatch nothing and return `Ok`. All three
//! refusals `serve::logits` makes before it reads a byte were made against
//! nothing. And deleting the SPIR-V walk's zero-length refusal does not fail
//! the suite; it HANGS it, which is what a corrupt module would do to a
//! driver.
//!
//! Three survivors are recorded rather than fixed, each with its reason at
//! the line: `dims_of`'s `in_width` has no consumer in any text here,
//! `plan_one`'s empty-grid refusal is unreachable from a real plan and is
//! witnessed by `tests/device.rs` alone, and `Bound::at`'s alignment clamp
//! needs a driver that reports zero alignment.
//!
//! # What is not here
//!
//! Weights, past a store that holds bytes under a name. Nothing in `src/`
//! loads a checkpoint. `Arg::Weight` carries a name and no WIDTH, so a *plan*
//! does not say how large a tensor is, and most whole-plan tests hold one
//! four-megabyte block under all 704 names -- which computes plumbing rather
//! than arithmetic.
//!
//! **One of them does not, any more.** A load plan states a width and a source
//! span, and `model_loader::executor::Execution` runs one. So
//! `tests/device.rs` executes the load plan for `mlx-community/Qwen3-0.6B-`
//! `4bit`, resolves all 704 names through [`names`], hands each its own tensor
//! at its own size, and fires. The distribution that comes back is this model's own,
//! and the two matmul kernels choose the same token on all sixteen rows.
//!
//! **And the model answers.** Shown an arbitrary six-token sequence five
//! times, `driver-vulkan` continues it: prefill 32 rows, then decode four
//! tokens one at a time, and all four are the rest of the cycle. That is
//! induction, the first circuit a language model learns, and it is a claim
//! ACROSS positions -- so it fails if the KV paging, the positions the book
//! hands each step, or the prefill and decode plans agreeing are wrong, none
//! of which a single fire can check. Zero weights answer `151935` four times
//! instead. A wrong rotary theta does NOT break it, which is recorded at the
//! test as a real limit on what it proves.
//!
//! That fire found the crate's own instrument to be wrong. The cross-check
//! measures a per-element RELATIVE difference and reports 1.99 on real
//! weights, which reads like a broken kernel; the largest ABSOLUTE
//! disagreement is 0.469 on a distribution spanning 40.4, and the ratio is
//! only large because 393_717 logits sit near zero. A tiled GEMM reduces 1024
//! bf16 terms in a different order than a vector kernel does, and eight bits
//! of mantissa cost a few tenths at a magnitude of twenty. Invented weights
//! hid it entirely -- every packed block held one repeating pattern, so every
//! partial sum was the same size and the order stopped mattering.
//!
//! For qwen3-0.6B the remaining widths are safe rather than merely untested:
//! exactly three names -- `embed` and its two sidecars, 77_791_232 bytes for
//! the packed half -- exceed the four-megabyte block, out of 335_372_288 for
//! the whole model. Which matters because this card returns ZERO for an
//! out-of-bounds storage read, silently, with the validation layer saying
//! nothing.
//!
//! # A second model, and the first oracle
//!
//! One checkpoint cannot tell a driver from a driver that fits one model, so
//! `mlx-community/Qwen2.5-1.5B-Instruct-4bit` is served too: two kv heads
//! rather than eight, a fused qkv projection, no qk-norm, an mlp of 8960, 648
//! bound weights. Every name still resolves through [`names`] and nothing is
//! left over, which is what makes that table a conversion rather than one
//! model's spelling list.
//!
//! Three things it changed:
//!
//!   - **The four-megabyte block was one model's arithmetic.** Eighty-seven of
//!     qwen2.5's weights exceed it, not three: the mlp projections overflow in
//!     all 28 layers.
//!   - **The verbatim read was one model's plan.** qwen2.5's states 535
//!     `TileMap` transforms for its fused qkv, so the loader's own executor
//!     runs it now, which is what a real driver would do anyway.
//!   - **The model does not continue the pattern**, and the driver is not what
//!     is wrong. A numpy forward of the same checkpoint -- sharing no code with
//!     this crate -- continues it with the attention biases and reproduces the
//!     card's answer token for token WITHOUT them. `LlamaLikeFacts` states
//!     `qkv_bias: true` for this model and the Metal text ignores the fact: no
//!     plan it lowers binds a bias, so this driver is asked to compute a Qwen2
//!     without one and does. The fix belongs in `crates/model`; a driver
//!     inventing weight names would be a driver deciding what a model computes.
//!
//! That reference is also the first independent oracle any number here has
//! been held against, and it agrees with the card on qwen3-0.6B as well.
//!
//! # The whole distribution, not just its argmax
//!
//! Which token won is the weakest claim a distribution can make: a row with
//! the right peak and a noisy tail decodes greedily forever and samples like
//! nothing. So both models' rows are held against that reference as NUMBERS --
//! the eight highest logits at the ids it ranked, five probes spread across the
//! vocabulary, and the row's range.
//!
//! They agree within 0.5, which is six bf16 steps at this magnitude and about
//! 2% of either row's range; the measured gaps are 0.06 on qwen3 and 0.4 on
//! qwen2.5, the wider one being the model that sums more terms per output. The
//! ranking is compared as a SET and only for seven of eight, because ranks four
//! through six on qwen2.5 sit within three bf16 steps of each other and their
//! order is not information. Two controls: a row moved by slightly more than
//! the tolerance must fail, and each model's row must fail the other model's
//! reference.
//!
//! That loader is not missing work in this crate, which is worth stating
//! because it looks like it should be. `tests/checkpoint.rs` measured a real
//! `Qwen/Qwen3-0.6B` snapshot against a real qwen3 plan: ZERO of 704 weight
//! names agree. The plan says `layer.0.down` where the checkpoint says
//! `model.layers.0.mlp.down_proj.weight`, and the plan wants `embed.scales`
//! and `embed.zeros`, which no bfloat16 checkpoint holds under any spelling
//! because they are outputs of quantizing. Loading is therefore a CONVERSION,
//! it already has a home in `model-loader`, and what this crate owes is
//! exactly `Weights::hold` -- a name, some bytes, and no opinion about where
//! they came from. Running that conversion and comparing again still leaves
//! 704 of 704 disagreeing, because the remaining gap is a naming convention;
//! [`names`] closes it.
//!
//! A sampler OF ITS OWN. [`turns::Serving::step`] drives fire after fire and
//! [`serve::logits`] names where the distribution is, but no code in this
//! crate chooses a token from it. That is deliberate and matches
//! `driver-metal`: sampling is policy -- temperature, top-p, penalties, a
//! grammar -- and a driver that held one would be a driver a server had to
//! argue with.
//!
//! What it does do is RUN the sampler the engine sent. A sampler on this
//! plane is a PTIR program, and [`frames::run_programs`] fires each of a
//! frame's instances over its own request's distribution, which is how the
//! answer reaches the channel the engine reads. The distinction is between
//! holding a policy and executing one.
//!
//! # The channel plane costs nothing, once its edge is narrowed
//!
//! Five of the engine's fourteen verbs are registration -- programs, channels,
//! instances, and closing the last two -- and none of them touches a device.
//! [`programs`] serves all five, and nearly all of its body is the conversion
//! between the ABI's records and the `driver` crate's, which already owns the
//! plane for the CUDA and Metal shells. The sixth verb is FIRING an instance,
//! which is that crate's reference interpreter run on the host over the rows
//! of the read-out that instance owns -- see [`frames::run_programs`] for why
//! "its own rows" is the load-bearing half of that sentence.
//!
//! Taking that crate as a dependency nearly broke the rule this driver is
//! built to keep. `driver` names no `tarpc` anywhere in `src/`, but its edge
//! took `driver-api`'s DEFAULT features, and `rpc` reaches `js-sys` and
//! `wasm-bindgen-shared` through tokio -- two crates `tests/pure.rs` refuses.
//! This crate's own edge to `driver-api` had already been narrowed for exactly
//! that reason; the new one had not, so the guard caught a closure widening
//! that no code in this crate could see. `driver`'s edge is
//! `default-features = false` now, which is a fact about a crate that never
//! wanted the feature rather than a concession made for this one.
//!
//! # What a frame may name, and what is refused by name
//!
//! A `LaunchPlan` carries far more than this driver implements. Eight of its
//! features are refused at admission rather than dropped -- recurrent state,
//! a user mask, `max_layers`, `hook_page_mask`, `dense_device_mask`, images,
//! audio and pre-embedded rows -- and [`frames::unserved_in`] is the list.
//!
//! Refusing by PRESENCE is what that list originally did, and for three of the
//! eight it was the wrong reading. The multimodal side-channels are CSRs with
//! one boundary per request, so a text-only batch of two carries `[0, 0, 0]`
//! in each of them: present, and naming nothing. And the engine SYNTHESISES a
//! causal mask -- `all_true(pos + 1)` -- for every request in a frame that
//! mixes a prefill with a device-resolved decode, because the bridge's mask
//! view is one flattened row per query row. Both refusals fired on frames that
//! asked for nothing this driver cannot do, and neither could be seen from a
//! single request. What is read now is the CONTENT: a CSR's last boundary, and
//! a mask's runs against the position of the query it belongs to. A sliding
//! window and a mask with a hole are still refused, because dropping a real
//! restriction silently is the failure this list exists to prevent.
//!
//! The reason each one is refused rather than ignored is that none of them
//! fails loudly when dropped. A frame whose `max_layers` is ignored runs the
//! full depth; one whose `hook_page_mask` is ignored reads the pages the
//! scheduler substituted away from; one whose `image_pixels` are ignored
//! answers a prompt whose picture vanished. Every one of those answers is
//! finite, fluent and wrong, which is exactly the class of failure no
//! validation layer and no assertion catches.
//!
//! # Two page allocators, and the one a frame uses
//!
//! [`pages::Book`] is this driver's own allocator and the right one for a
//! server built on this crate alone. The ENGINE has its own -- it runs
//! eviction, prefix sharing and the copy plans -- and hands down a
//! `kv_page_indices` CSR. Two allocators handing out page 7 is not an error
//! anybody sees: attention reads another conversation's keys and the model
//! stays fluent.
//!
//! **That CSR does not name the pages it looks like it names.** Its entries
//! are the conversation's OWN working set -- page 0 is the first page of this
//! conversation -- and `FrameSubmission::kv_translation`, partitioned per
//! roster row, says which pool page each one was placed in. [`envelope`] is
//! where the two are joined, and the reason it is a module rather than a line
//! is that skipping the translation is invisible to every gate a single
//! conversation can pass: an identity map onto itself is consistent with
//! itself. It took two conversations at once to see it, and what it looks like
//! is the second one answering the first one's question fluently.
//!
//! So [`shell::Shell::launch`] does not touch the book at all. [`frames`]
//! splits a frame's CSRs into this driver's [`resources::Request`]s and
//! [`turns::Serving::over`] fires them, which is [`turns::Serving::step`]
//! minus the growth. Measured on a real qwen3: one conversation served both
//! ways, BIT FOR BIT the same distribution, with a control proving the
//! frame's pages are what attention actually reads.
//!
//! Two findings came out of that measurement:
//!
//!   - **A single-request frame cannot see a per-request page CSR.** With one
//!     request, its span IS the whole page list, so a conversion that ignored
//!     `kv_page_indptr` entirely passed every assertion. The device test fires
//!     a TWO-request frame for that reason, and the mutation is then caught by
//!     `Frame::of`'s aliasing check rather than by a number.
//!   - **`Pool::resize` could kill the process.** Its staging buffer is a
//!     plain `vec![0u8; bytes]`, and a `Vec` that cannot be allocated aborts
//!     rather than returning. Mutating `launch`'s admission to admit
//!     everything asked for seventy terabytes a layer and the test binary died
//!     with SIGABRT. The admission check is still where a scheduler gets its
//!     answer; `resize` now asks with `try_reserve_exact` first, because a
//!     pool a caller's arithmetic slip can kill is not one a server can be
//!     built on.
//!
//! What [`programs`] runs is a program's HOST stages, and it runs them when
//! the composer's decision still holds. [`programs::Programs::fire`] is
//! `driver`'s reference pass and [`frames::run_programs`] is the loop the
//! engine seam drives it from, each member first checked against the ring
//! words the scheduler pinned when it composed the batch -- so a fire that
//! arrives after another moved the ring is skipped and re-posted rather than
//! answering into somebody else's cell. What is still NOT run is any program
//! stage on the DEVICE: this driver advertises no codegen backend, holds the
//! emitted kernels a registration carries, and runs the stages the
//! interpreter can. That is the shape `driver-metal` serves too.
//!
//! # A decode that states none of its own geometry
//!
//! Every inferlet in this tree decodes with a device-carried loop, and such a
//! step arrives with its geometry ELIDED: one placeholder token, one
//! placeholder position, and empty page tables. The tokens do not exist when
//! the frame is sealed, because they are what the PREVIOUS step's program put
//! on a channel. The engine offers this shape as `GeometryClass::
//! DecodeEnvelope`, and only to a driver that advertises the ports it will
//! resolve.
//!
//! [`envelope::fill`] is that resolution: host members are copied through the
//! page translation, envelope members take their token and position from the
//! channels the last fire wrote, and the page span follows from the position.
//! It costs one invariant, and the invariant is worth naming because it was
//! deliberate: a frame of envelope steps CANNOT be converted whole and then
//! fired, since step n+1 does not exist until step n has run. The engine seam
//! drives those one at a time; [`shell::Shell::launch`] keeps the stronger
//! order for a frame of ordinary host-wire steps, which is why it splits into
//! `admit`, `prepare` and `serve`.
//!
//! A channel whose producer has not run yet is `Filled::Early` -- not a fault
//! and not a fire. Its members are published as RETRY and the scheduler posts
//! them again. Instrumenting a real `pie serve` never reached it, because the
//! program that fills a channel runs in the same call that reads it; it is
//! kept because `NotReady` is a state the resolver can return, and answering
//! it with a fire would sample a distribution nobody computed.

// The manifest deliberately does not take the workspace lint table, because it
// forbids `unsafe_code` and every `ash` entry point is unsafe. The rest of that
// table is worth having, so it is restated here without that one name -- and
// the portable half keeps its own guarantee a different way, by containing no
// `unsafe` at all, which `tests/pure.rs` asserts by reading the modules this
// file does not gate.
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout)]

#[cfg(feature = "native")]
pub mod binding;
#[cfg(feature = "native")]
pub mod device;
#[cfg(feature = "native")]
pub mod dispatch;
// A step's geometry, filled in from the channels the last fire wrote it into
// and translated from working-set pages to physical ones. Gated with the
// serving half because it reads the program registry, though it is arithmetic
// over CSRs and needs no device.
#[cfg(feature = "native")]
pub mod envelope;
// Two constants and a function over a `Device`. Ungated so that
// `facts::PAGE_SIZE` and `facts::BACKEND` -- which an engine reads to CHOOSE a
// backend, before it has one -- do not require Vulkan to be present.
pub mod facts;
// The engine's frame, split into this driver's requests. Gated with the rest
// of the serving half because `Request` and `Step` are, though the split
// itself is arithmetic over CSRs and its tests need no device.
#[cfg(feature = "native")]
pub mod frames;
pub mod geometry;
pub mod lowering;
// Strings and a static table, so it is ungated: a caller that only wants to
// know what a checkpoint calls `layer.3.down` should not have to link Vulkan.
pub mod names;
// Pure Rust, and gated only because it speaks in `resources`' `Shape` and
// `Request` -- which are pure data in a module that also holds device
// handles. Splitting that module to ungate this one would buy nothing today.
#[cfg(feature = "native")]
pub mod pages;
#[cfg(feature = "native")]
pub mod resources;
// Ungated on purpose: the channel plane needs no device, which is the whole
// argument for the `driver` crate existing, and gating it would make the
// no-default-features build unable to state a verb it can serve in full.
pub mod programs;
pub mod rope;
#[cfg(feature = "native")]
pub mod serve;
#[cfg(feature = "native")]
pub mod shell;
pub mod spirv;
#[cfg(feature = "native")]
pub mod turns;

#[cfg(feature = "native")]
pub use binding::{Arena, Resolve, Unbindable, bind, resolve};
pub use geometry::{Dims, Local, Module, Rule, Tile, Ungeometric, groups, lanes};
pub use lowering::{Call, Mismatch, Value, pack};
pub use spirv::Declared;
