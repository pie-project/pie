//! The Vulkan execution shell: instances, queues, descriptor pools, command
//! buffers, and the arithmetic that turns one rectangle of a plan into one
//! `vkCmdDispatch`. `kernels-vulkan` owns the SPIR-V modules and the table of
//! what their entrypoints take; this crate owns what it takes to fire them.
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
//! decided when `slangc` ran. So the driver's arithmetic is a division by a
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
//! and every layer; [`device::Device::copy_within`] is the copy under it, on
//! the copy engine because a host `memmove` of write-combined VRAM costs
//! eighteen times what `vkCmdCopyBuffer` does. Only
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
//! Both mixtures of experts, at a 64-row prefill, did not -- and neither did
//! two runs of the ONE-AT-A-TIME reference against each other. This crate
//! read that as a property of the kernel: `route_sort` builds its permutation
//! with workgroup-scoped atomics, so when two rows want the same expert the
//! order is whichever lane won, and the gather then writes the same rows to
//! different offsets. A decode is one row and has nothing to tie, which is
//! why nothing here saw it until prefills were fired. The conclusion drawn
//! was that a driver cannot fix this and should not paper over it.
//!
//! **That was wrong, and it is worth leaving the mistake visible.** The
//! varying arena was not a benign permutation, it was a memory race:
//! `route_sort`'s `n` was `n_experts * k` where `expert_ids` holds
//! `tokens * k`, so the kernel scanned 112 entries past its own region into
//! the `perm` it was concurrently writing. Three sibling launches were wrong
//! in the same way. Upstream found it from the Metal side, where the same
//! fire produced 0 NaNs, then 6, then 208004, and fixed all four.
//!
//! **And then the correction was wrong too.** With the race fixed, the test
//! was inverted to demand byte-equal arenas, and it passed three times. It
//! was luck. The arenas of two routed prefills differ in 75k to 198k of 6.2M
//! bytes, and 141 bytes of `route_sort`'s own 512-byte `perm` region are
//! among them -- INSIDE the buffer the router owns, which is what separates
//! an ordering from an overrun. The first reading had been right about the
//! atomics after all. The race and the ordering were two different things
//! wearing one symptom, and each correction threw out the other's half.
//!
//! So the lesson is not "an arena that varies is a race", which is what this
//! crate wrote down in between. It is that a symptom can have two causes, and
//! that fixing the one you found does not retire the one you did not. What a
//! driver author gets from a varying arena is a question, not a verdict --
//! and the way to answer it is to ask WHERE it varies, because a difference
//! inside a buffer's own region and a difference past its end are different
//! findings that a byte-position comparison reports identically.
//!
//! The test now claims what the ordering cannot move. Two fires of one routed
//! prefill choose the same token on every row -- an argmax per row, measured
//! at 0 flips of 16 -- and their distributions stay within a bound, measured
//! at 0.0067 against an allowed 0.05. The arena is not compared at all,
//! because there is no arena claim here that is both true and worth making:
//! `whole_plan` already re-runs its reference against itself precisely so it
//! can tell "the batching is wrong" from "the plan has no single answer".
//!
//! The whole-plan test still runs the reference a second time before it
//! accuses the batching. That guard cost one run and would have named the
//! kernel rather than the command buffer, which is what it did.
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
//!   - **The model did not continue the pattern, and the fix was not here.** A
//!     numpy forward of the same checkpoint -- sharing no code with this crate
//!     -- reproduced the card's answer token for token WITHOUT the attention
//!     biases. `LlamaLikeFacts` stated `qkv_bias: true` and the Metal text
//!     ignored the fact: no plan it lowered bound a bias, so this driver was
//!     asked to compute a Qwen2 without one and did. That belonged in
//!     `crates/model` -- a driver inventing weight names would be a driver
//!     deciding what a model computes -- and it was closed there: an
//!     `add_bias` row in both kernel crates, an `add_bias` fact through the
//!     text, and `q_bias`/`k_bias`/`v_bias` in [`names`]. The oracle was
//!     regenerated with the biases and this driver now matches it.
//!
//! That reference is also the first independent oracle any number here has
//! been held against, and it agrees with the card on qwen3-0.6B as well.
//!
//! The bias is now gated end to end as well as numerically, by
//! `vulkan_second_model` in `tests/gpu`, and the gate was checked by breaking
//! what it claims to cover: zeroing every `_bias` tensor as it is held zeroes
//! 84 tensors under Qwen2 and turns the answer into noise, and zeroes NONE
//! under Qwen3 and changes nothing. That asymmetry is why serving a second
//! architecture is a gate rather than a nicety -- a whole-stack test on
//! Qwen3-0.6B cannot fail when the bias path breaks, however it is written.
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
//! Running one is how this crate found that nobody had. Every temperature but
//! zero panicked in the shared interpreter -- `pivot_threshold`'s predicate
//! payload does not ride in the op's `args`, and the walk that builds a
//! stage's index only read `args`, so the `p` of a nucleus was an operand
//! nothing evaluated. The panic killed the driver thread with a frame in
//! flight, which is worse than a failed request: the engine waited for a
//! launch that would never complete, and the stall line said
//! `driver 0 stalled for 7030.132606596s`. Fixed in `crates/driver`, not
//! here, and gated by `vulkan_sampled_completion`, which fires two sampled
//! completions at a real device because nothing else in the tree does.
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
//! That plane is now proved in all three of the directions a guest can drive
//! it, each by its own gate under `tests/gpu/`. `vulkan_programmable_sampler`
//! reads the distribution OUT and decides the token host-side
//! (`mirostat-v2-sampling`, whose `mu` moves only if a real surprise came
//! back). `vulkan_sampling_primitives` keeps the whole decision INSIDE one
//! epilogue and reads six channels out of it, three of them 151936 wide.
//! `vulkan_grammar_constrained` writes INTO the epilogue every step: a host
//! grammar matcher puts its allowed-token mask into a channel that
//! `masked_argmax` takes as an OPERAND, and puts a different one on the next
//! fire of the same instance. That last one is the only shape where a bug is
//! invisible from the outside -- a mask bound once and reused decodes
//! perfectly well, just against a stale grammar -- so its control is a schema
//! whose required keys are `zqx` and `wbn`, words no continuation of the
//! prompt would choose. The guest also checks our work for us there:
//! `accept_tokens` fails on any token the matcher did not allow, so a run
//! that returns AT ALL is a run in which every token was legal under the mask
//! standing when it was chosen.
//!
//! What that gate deliberately does not claim is that a schema terminates.
//! The inferlet decodes greedily and a JSON grammar permits unbounded
//! whitespace before a closing brace, so a single-property object reaches
//! `{"age": 24` and then emits `\n  \t\t\t` forever, every repetition legal;
//! the inferlet's own default schema enumerates `skills` past 256 tokens for
//! the same reason. Both were measured here and both are guest facts, not
//! driver ones.
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
//! Reading the content was only HALF done, and the other half cost three
//! inferlets. `has_user_mask` stayed the first half of the condition, so the
//! rows a GUEST named were still refused unread -- the content check only ever
//! saw the masks the engine synthesised. And the row check compared a row's
//! WIDTH to the query's position, which is not what causal means. A guest
//! builds its mask as `[queries, pool_len]`, because that is the shape a pool
//! has, so row 0 arrives as `runs=[0, 1, 47]` over a width of 48: one true
//! cell, forty-seven false ones. Causal, and refused for being a rectangle.
//! It was measured coming off the wire from `contrastive-decoding` with its
//! window opened past the pool, where the guest's own mask reduces to
//! `key <= query` exactly.
//!
//! What is checked now is the TRUE SET, clamped the way `EncodedMask::
//! expand_into` clamps: the true cells must be exactly `0..=position`. Two
//! rules say that -- no true run reaching past the diagonal, and the diagonal
//! reached -- and three others that stood there did not survive. Coverage of
//! `total_size` refused a row that names its true cells and leaves its false
//! tail unspoken, which the encoding defines as false anyway; a ban on false
//! runs before the diagonal, and a minimum width, are both implied by the two
//! once the runs are clamped. Each was removed for the same reason: no
//! mutation of it could be made to fail a test, which is the definition of a
//! claim this crate cannot check.
//!
//! `tests/gpu/tests/vulkan_padded_causal_mask.rs` holds both answers from one
//! inferlet and one parameter: with the window open, `contrastive-decoding`
//! runs and answers; at its default of 8, the same run is refused by name.
//! The refusal half is the control, because a fix that simply stopped
//! checking masks would pass the other one and attend outside its window
//! fluently, with nothing said.
//!
//! # What this backend actually serves, counted
//!
//! The mask defect above was found by RUNNING things rather than by reading
//! them, so the rest of `tests/inferlets` was run too -- thirty of them,
//! against a real `pie` on this driver. What came back is a coverage map, and
//! it is worth writing down because "the driver works" is not a measurement:
//!
//! * **Twenty answer.** Every sampler-shaped one: `naive-baseline`,
//!   `repetition-penalty`, `dry-repetition-penalty`, `top-a-sampling`,
//!   `xtc-sampling`, `tail-free-sampling`, `locally-typical-sampling`,
//!   `eta-epsilon-sampling`, `entropy-adaptive-temperature`,
//!   `mirostat-v2-sampling`, `sampling-primitives`, `gumbel-watermark`,
//!   `greenlist-watermarking`, `synthid-tournament-sampling`,
//!   `classifier-free-guidance`, `context-aware-decoding`, `token-healing`,
//!   `tart-masked`, `asap-grammar-aligned-decoding`,
//!   `json-schema-constrained-decoding`. Five of those now have gates.
//! * **Six are refused by a MODEL profile**, not by this driver, and
//!   `driver-metal` refuses them identically -- it advertises the same
//!   `has_attn_score: false`, `has_lora: false`, `has_mtp_logits: false`,
//!   `has_kv_envelopes: false`. `quest-attention`, `tova-attention`,
//!   `trackb-h2o`, `trackb-snapkv`, `mtp-speculative-decoding` and
//!   `lora-probe` are parity, not gap.
//! * **Four stop at an ENGINE wall** this driver never sees:
//!   `EmbedTokens is not host-derivable: channel N has no host-known value`,
//!   reached after the engine reports that `a channel-bound dense AttnMask
//!   belongs to the pool-owned device-geometry class` and falls back to
//!   host-evaluated execution. `beam-search`, `naive-masked`,
//!   `sliding-window-attention` and `attention-sink`. The last two only
//!   reach it because the mask fix above let them past this driver at all.
//! * **Two fault inside the reference interpreter** with a read-out this
//!   driver reads one row of. See the note in `frames::unserved_in` where the
//!   multi-row read-out refusal used to stand: the guest
//!   sees `driver published poison epoch 1`, which names nothing, and the
//!   reason is only in the engine's WARN log.
//!
//! So of the failures, none is a Vulkan kernel, none is a binding, and none
//! is this driver's lowering. That is the claim the sweep was worth making,
//! and it is a different claim from any test passing.
//!
//! ## The same sweep again, after the barriers were narrowed
//!
//! Dropping 140 of 451 barriers per step is the one change in this crate whose
//! failure mode is a RACE, and a race does not fail a unit test reliably. The
//! device tests pin it two ways (a barrier-count range, and a byte-for-byte
//! comparison against a one-dispatch-per-submit recording), but both run the
//! same handful of plans. So the whole curated harness was re-run afterwards,
//! against a `pie_server` wheel built `--no-default-features --features
//! driver-vulkan`, on `Qwen/Qwen3-0.6B-optimized`:
//!
//! ```text
//! 24/39 passed
//! ```
//!
//! Thirty-nine now rather than thirty, because the harness has grown. Nothing
//! that answered before stopped answering, and twenty-four whole inferlets
//! producing their expected text end to end is the strongest available
//! evidence that the hazard analysis did not drop a real dependency.
//!
//! The interesting half is the fifteen that do not, because this time every
//! one of them was run down to a NAMED cause rather than sorted into a bucket:
//!
//! * **Eight** hit the `EmbedTokens is not host-derivable` engine wall:
//!   `beam-search` and its two variants, `attention-sink` and
//!   `sliding-window-attention` and their `-attends-prompt` variants, and
//!   `consensus-decoding`.
//! * **Four** are model-profile refusals `driver-metal` makes identically:
//!   `quest-`, `tova-`, `h2o-` and `snapkv-attention`.
//! * **`prefix-tree-kv-cache`** is the guest pipeline-contract problem
//!   dissected in `engine::driver::backend::vulkan` -- one leaf's `run_ahead`
//!   closes a pipeline its three siblings still need.
//! * **`contrastive-decoding`** is refused BY NAME, and the name is this
//!   driver's: `this driver does not serve a user mask: attention here is
//!   causal, and a plan's mask would be dropped rather than applied`. That is
//!   the refusal `tests/gpu/tests/vulkan_padded_causal_mask.rs` exists to pin
//!   from both sides, so the sweep is re-measuring a gate, not finding a bug.
//! * **`constrained-speculative-decoding`** does not close its JSON. This one
//!   was the only genuinely open question, and it took two experiments.
//!   Running the arms separately shows the failure is in the `draft_length=0`
//!   arm -- plain grammar-masked greedy decoding, with the drafter
//!   short-circuited -- so speculation is not implicated at all. Raising the
//!   budget from 256 to 1024 tokens does not close it either, so it is not a
//!   budget. The control that settles it is a DIFFERENT BACKEND on the same
//!   box and the same model: a wheel built `--features driver-wgpu` fails the
//!   same arm with the same message. A defect two independent backends
//!   reproduce is not in either of them.
//!
//! So of the fifteen, none is a Vulkan kernel, none is a binding, and none is
//! this driver's lowering. That is a stronger claim than the earlier sweep
//! made, and it is stronger because the sixteenth failure stopped being one:
//! `cacheback-speculative-decoding` now PASSES, on the multi-row read-out work
//! recorded beside `Frame::sampling_indptr`.
//!
//! ### What that list looks like now: 35 of 39
//!
//! The paragraphs above are the record of ONE sweep, and every line of it has
//! since been overtaken; they are kept because the reasoning is what earned
//! the next three findings, not because the counts still hold. As measured
//! now, on the same box and model:
//!
//! * The **eight** `EmbedTokens is not host-derivable` failures are gone.
//!   They were one wall wearing eight faces: a channel-bound dense `AttnMask`
//!   declined to the device-geometry class, so the pass fell back to a host
//!   evaluation that cannot derive an embedding. Serving that class -- pages,
//!   page CSR, write descriptor, mask rectangle -- passes all eight.
//! * **`contrastive-decoding`** passed with them. The refusal it was pinned
//!   against is the one that work removed, so the gate
//!   `tests/gpu/tests/vulkan_padded_causal_mask.rs` now pins the SERVING, and
//!   its narrow-window control moved into a mutation-tested unit test because
//!   the inferlet's text turned out to be insensitive to the window.
//! * **`prefix-tree-kv-cache`** was never a contract problem. Its guest had
//!   already been rewritten to one pipeline per leaf; the harness installs
//!   PREBUILT `.wasm` and never built it, so the old shape kept being tested
//!   and its `pipeline is closed` kept being read as an engine wall. A probe
//!   on the host's `pipeline::new` settled it in one run -- five guest
//!   `Pipeline::new()` calls, one host pipeline. `tests/inferlets/conftest.py`
//!   now builds the guests it tests.
//! * **`constrained-speculative-decoding`** was not a defect either, and the
//!   earlier two-backend argument above was right about that much and wrong
//!   about where to look. Run with a bounded schema, both arms return the
//!   IDENTICAL 37 tokens and speculation cuts 37 forward passes to 33 -- the
//!   exact property the example exists to demonstrate. Unbounded, the model
//!   writes a coherent, schema-valid skills list past 768 tokens and the
//!   example fails on its own cap. The schema now carries `maxItems`.
//! * The **four** model-profile refusals are unchanged, and are the whole
//!   remainder: `quest-`, `tova-`, `h2o-` and `snapkv-attention` want
//!   `attn_score` or `envelope_dot`, which this model does not gate on and
//!   `driver-metal` refuses identically.
//!
//! Two of those four findings were in the harness and the example rather than
//! in any driver, and both had been attributed to a driver for weeks. A sweep
//! is only evidence about the thing under test if the thing under test is
//! what was built.
//!
//! ### The log that was not being written
//!
//! Every "and no driver fault" above was unfalsifiable until a fix elsewhere.
//! The pyo3 wheel installed no `tracing` subscriber at all, so every
//! engine-side `warn!` went nowhere and `RUST_LOG` appeared to do nothing.
//! What reached Python was the terse end of a poisoned channel -- `driver
//! published poison epoch 1`, a WORD with nowhere to put a sentence -- which
//! is exactly the read-out this crate complains about elsewhere.
//!
//! `sdk/server/python` now `try_init`s a stderr subscriber on the CLI's
//! conventions. It paid for itself immediately: the first run under it named
//! `logits intrinsic row range exceeds the forward's readout rows`, which is
//! the fault the multi-row read-out work was aimed at, and the second named
//! the user-mask refusal above. Neither was reachable from the harness before.
//!
//! # A row number that meant two different things
//!
//! The multi-row read-out work above shipped with a bug that no test in this
//! crate could have caught, and the shape of the blind spot is worth more
//! than the fix.
//!
//! `envelope::fill` read `sampling_indices` as rows of the whole FIRE and
//! rebased each through its request's start: `base + (row - lo)`. The engine
//! states them per REQUEST. `scheduler::wire` appends each request's indices
//! unchanged and asserts `index < row_len` on the way past;
//! `driver::resolve` writes `span - 1` when no read-out port is bound, under
//! a test whose message is "both relative to their own request". Two
//! independent statements of the same convention, and this driver read the
//! other one.
//!
//! Subtracting `lo` is the IDENTITY when `lo == 0`. Every single-request plan
//! has `lo == 0`, and every unit test in this crate builds a single-request
//! plan, so a hundred and twenty-six of them agreed with a wrong reading.
//! One conversation could never see it either -- its own numbering and the
//! fire's are the same numbering.
//!
//! What saw it was eight conversations at once. `vulkan_many_conversations`
//! refused its own fire:
//!
//! ```text
//! request 1 reads out row 91, which is not in its own rows 92..184
//! ```
//!
//! 91 is request 1's own last row, and also request 0's. The driver was
//! reading a correct plan as a corrupt one, and the failure was LOUD -- but
//! only because the bounds check happened to be there. Without it the fire
//! would have been servable and request 1 would have read request 0's
//! distribution: a second conversation answering fluently in the first one's
//! context, which is the failure this crate keeps finding at every layer.
//!
//! The half that took the longest was deciding WHERE to fix it, and the
//! first answer -- "the envelope is the seam where the numbering changes;
//! the fix belongs there and nowhere else" -- was wrong. It survived because
//! `frames::sample_rows_of` seemed right to read the other way: `frames`
//! reads what `envelope` writes, so if the envelope flattened, the frame
//! should rebase. That reasoning holds only if the envelope is the sole
//! producer, and it is not.
//!
//! `envelope::fill` has no caller inside this crate. The engine calls it and
//! then calls `Shell::prepare`; `Shell::launch` -- the entry these device
//! tests use -- calls `prepare` on the RAW wire plan. So `sample_rows_of`
//! had two producers stating two different conventions, which is exactly why
//! fixing "one place" kept moving the failure to the other. The batched
//! host-wire case in `a_frame_the_engine_built_answers_what_the_driver_s_own
//! _turns_do` is what made that visible, by taking the path that skips the
//! envelope entirely.
//!
//! The fix is one convention end to end: `sampling_indices` are numbered
//! within their own request from the engine's geometry to
//! `resources::Request::samples`. `fill` no longer flattens and
//! `sample_rows_of` is the identity plus a width check. The clearest
//! evidence the old reading was wrong is that the pre-existing batched
//! device fixture had `2 * len - 1` written into its own plan -- a number
//! the engine never produces, authored to match the driver rather than the
//! contract.
//!
//! Pinned from both sides: `envelope::tests::a_read_out_row_is_numbered_
//! within_its_own_request` and `frames`' namesake are MULTI-request for the
//! reason above, and restoring either the `base +` or the `- lo` fails the
//! device test and `vulkan_many_conversations` -- the latter being the only
//! gate that runs the engine's own path.
//!
//! ## The same blind spot, one branch over
//!
//! Having named the shape -- a per-request quantity that a single request
//! cannot distinguish from a per-fire one -- it was worth asking where else
//! `fill` had it, and the answer was the branch immediately above.
//!
//! A device-resolved decode envelope carried NO read-out table at all and
//! leaned on `Request::read`, which answers with the last row when the table
//! is empty. For a one-row decode that is the same list, so the fixture that
//! tested it could not tell the two rules apart. `driver::resolve` states
//! `span - 1` per request and a bound read-out port states whatever the
//! program asked for, so a SPECULATIVE decode -- three rows, reading rows 0
//! and 2 -- resolved two rows and was served one.
//!
//! That is the same fault the host-wire branch produced when its table was
//! dropped, and it announces itself the same way: `logits intrinsic row
//! range exceeds the forward's readout rows`. Both branches now carry the
//! table, in the envelope's numbering, through the same conversion.
//!
//! The test that pins it had to be built rather than adjusted: the existing
//! envelope fixture is one token on capacity-one channels, and one row is
//! exactly the case that cannot see this. `speculative_program` adds the
//! read-out port and widens the channels, which is the smallest fixture in
//! which "all the rows it asked for" and "the last row" are different
//! answers.
//!
//! ## A third site, and a guard that could not fire where it was needed
//!
//! Having fixed the numbering twice, the crate was swept for every other
//! place a per-request row is rebased into fire order. There are three.
//! `resources::Frame::of` is the legitimate one -- it is where request-local
//! becomes fire-relative, and it bounds-checks the row against the request's
//! own length first, refusing `NotItsRow`. `turns::spans_of` does the same
//! walk for the read-out spans, and did not check at all.
//!
//! The first thing written there was a comment explaining why no check was
//! needed: `over` reaches `spans_of` only after `tiled`, and `tiled` builds a
//! `Frame` per sub-fire, so `Frame::of` has already refused any bad row.
//! That explanation was then tested rather than trusted, and it is false for
//! exactly the case that matters. `turns::slice` keeps the samples landing
//! in its own piece (`row >= from && row < to`), so a row past the request's
//! end lands in NO piece, is filtered away, and every piece falls back to
//! "the last row". No `Frame::of` is ever handed the row. `over` then walks
//! the ORIGINAL requests. **The upstream guard cannot fire precisely when
//! this arithmetic is reached with a bad row**, which is to say the split
//! fire -- the path a long prompt takes.
//!
//! Measured before fixing: two requests of two rows, the first naming row 2,
//! gives `[[2], [3]]`. Request 0's answer is request 1's first row. The row
//! exists, the read succeeds, and the text is fluent -- the same silent
//! cross-request read as the two above, reached a different way.
//!
//! The check now sits with the arithmetic it protects, where a test reaches
//! it with no device: `a_read_out_row_past_its_own_request_is_refused_before
//! _it_is_renumbered`. Its companion,
//! `a_split_fire_filters_the_bad_row_away_instead_of_refusing_it`, pins the
//! reason the other guard was not enough, so the day `slice` starts refusing
//! instead of filtering, the record says what changed.
//!
//! Admission refuses these rows too -- `frames::sample_rows_of` and
//! `envelope::fill` both check the width -- so this is the third line of a
//! defence, not the only one. It is kept because it is the line that holds
//! when a `Request` is built inside this crate rather than parsed from a
//! plan, and because unlike the guard it replaces, it can be shown to fire.
//!
//! ## The fallback that was the defect it warned about
//!
//! `frames::member_requests` says which of a fire's requests a program
//! instance owns, and its doc records why it exists: `driver-metal` handed
//! every member the WHOLE read-out, the interpreter's `base_row` is 0, so in
//! a frame of three requests all three programs sampled the FIRST request's
//! distribution and returned its token -- one fire, three answers, all the
//! same, nothing faulting.
//!
//! It then fell back to `(0, requests)` -- the whole read-out -- for any CSR
//! it could not read, which is that defect. Measured on `[0, 1, 2, 9]` over
//! three requests: members 0 and 1 answer `(0, 1)` and `(1, 2)`, correctly,
//! and member 2 answers `(0, 3)`. One member of a frame silently answering
//! out of another conversation while its neighbours stay right, which is
//! harder to see than all three being wrong together. A roster longer than
//! the CSR describing it does the same.
//!
//! "Absent" and "present but not describing this member" are different
//! claims, and only the first means the single-member case. The second is a
//! frame whose tables disagree with its own roster, which both callers
//! already refuse by name a few lines earlier. `member_requests` now returns
//! `Option` and they refuse this too.
//!
//! Its doc had said this was "worth a named function and a test because
//! getting it wrong is invisible", and there was no test. There are three
//! now. The sentence was right about the invisibility and wrong about the
//! coverage, which is the same pairing as the guard above whose removal took
//! its own explanation with it.
//!
//! ## The same question, asked back
//!
//! A merge from `origin/rewrite` brought a new doc on
//! `driver_api::LaunchPlan::sampling_indices` stating the opposite of what
//! this crate had just concluded: "Absolute, not request-relative", with
//! `driver-metal`'s `rows_of` as the authority and a note telling the reader
//! not to take `scheduler::wire`'s assert -- the one cited here -- as a
//! contradiction.
//!
//! That is worth taking seriously rather than defending a position, so it was
//! measured again on the merged tree, with a dump in this crate's own
//! envelope and the 8-conversation gate unmodified:
//!
//! ```text
//! qo=[0, 92, 93]  sidx=[91, 0]
//! qo=[0,  1,  2]  sidx=[ 0, 0]
//! ```
//!
//! Request 1 spans rows 92..93 and names row 0. Absolute, that is request
//! 0's first row. The second line is two one-row requests both naming 0.
//! `driver::resolve` writes `span - 1` from the request's own row count, and
//! the engine emits nothing else today. The reading this crate uses is what
//! the engine produces, and the twelve gates pass on it.
//!
//! The other doc was not simply wrong, which is the useful part. It
//! accurately describes what `driver-metal` DOES -- and driver-metal is also
//! the backend that handed every program member the whole read-out, the
//! defect recorded two sections up. A backend reading absolutely and a
//! scheduler writing relatively agree exactly when `qo_indptr[lane]` is 0,
//! which is every single-request plan, which is why both readings survived
//! this long.
//!
//! One half is genuinely open and this crate should not claim otherwise:
//! when a guest binds `Port::Readout`, the values are the guest's own,
//! copied through unchecked, and a beam whose lanes are separate requests
//! naming its siblings' rows cannot be said relatively at all.
//!
//! The first version of that note said "nothing pins that branch -- no test,
//! no range check", and that was this crate making the same kind of claim it
//! had just corrected. `LaunchPlan::validate_geometry` DOES refuse a row past
//! the fire's total tokens, with a test, and that bound holds under both
//! readings. What is genuinely unchecked is narrower and more interesting:
//! that request `r`'s index falls within request `r`'s own span -- because
//! that check IS the relative reading, and putting it in the shared crate
//! would settle by fiat the one question the guest-bound branch has not
//! answered. This driver refuses that case at its own seam instead, which is
//! where a backend's reading belongs.
//!
//! ## The fifth instance, in the backend the doc had cited as the authority
//!
//! The shared doc's case for the absolute reading named a function:
//! `driver-metal`'s `lowering::frame::rows_of`. Reading it settled the
//! question in the other direction. It builds a `samples` flag per row by
//! `samples[row] = true` straight off the wire values -- absolute -- and
//! `model_compiler::lower::epilogue` filters exactly that flag to build
//! `logit_row_indices`, the gather. So it is not a grouping key with no
//! consequence; it is the read-out itself. On the live `qo=[0,92,93]
//! sidx=[91,0]` batch it marks request 0's FIRST token and leaves request
//! 1's only row dead, and the gather still finds two rows, so nothing
//! refuses.
//!
//! Its empty-table branch was wrong in the matching way -- `samples[n - 1]`,
//! the last row of the FIRE -- which in a two-request decode leaves the first
//! request with no readout at all. That is `envelope::fill`'s two branches
//! again, one backend over, and a test named
//! `a_step_naming_no_readout_reads_its_last_row` asserted `!rows[0].samples`
//! and so PINNED it. A fixture that cannot distinguish two rules is not a
//! test of either; this one had picked the wrong rule and held it.
//!
//! Both branches were corrected, a per-request `NotItsRow` refusal added
//! beside them, and -- because the translation now needs `sampling_indptr`
//! that seam had never read -- a `RaggedReadoutTable` refusal for a table
//! with values and no CSR to place them, rather than a fallback. That last
//! one is unreachable in production, and checked to be: `scheduler::wire`
//! seeds it `[0]` and pushes once per request, exactly as it does
//! `qo_indptr`. It exists to notice the day that stops being true.
//!
//! Which retires the disagreement rather than any crate's side of it. Three
//! backends now read the field one way, so the shared doc no longer records
//! two readings.
//!
//! ## The sixth instance, in the one place every backend shares
//!
//! `driver-cuda` was checked next and looked right: its empty branch says, in
//! as many words, "one row per REQUEST, `qo_indptr[r + 1] - 1`. Not the
//! fire's last row." Then it turned out to be an OVERRIDE, applied over the
//! answer of `model_compiler::lower::rows_from_regions` and only when the
//! table was empty. The shared helper it corrected reads the non-empty case
//! `samples[row]` -- absolute -- and CUDA's override could not see that,
//! because it only ran when there was nothing to see.
//!
//! So the sixth instance was the shared one, and it had been fixed halfway by
//! a caller. `rows_from_regions` now takes a `Readouts` -- indices, the CSR
//! that says whose they are, and `qo_indptr` -- because a caller that cannot
//! state all three cannot read the field, which is the whole failure said as
//! a signature. Both halves live there, with `NotItsRow` and
//! `RaggedReadout` beside them, and CUDA's override is deleted rather than
//! left half-right.
//!
//! And `driver-metal`'s `rows_of`, corrected an hour earlier, turned out to
//! be a line-for-line duplicate of that helper -- same region walk, same
//! tiling refusals, same flag. `Row`'s own doc had warned about exactly this
//! ("two chances for a bit to be read wrongly and no way to notice") and the
//! warning had already come true: both copies were wrong the same way, and
//! correcting one left the other. `rows_of` is now a call into the shared
//! rule with a `From<RegionDrift>` mapping that keeps that crate's
//! vocabulary, and its 185 host tests pass with the duplicated body deleted
//! -- which is the measurement that says the two copies really were one rule.
//!
//! The count of places this bug class was found stands at six: four here, one
//! in Metal's copy, and one in the code all of them share.
//!
//! ## Asking the shipped configuration, not the forced one
//!
//! `Fired::tiered` was added to make one test self-diagnosing, and the first
//! thing worth pointing it at was a question no test in this crate asked:
//! does the tile this driver actually SERVES reach the cooperative-matrix
//! build? `the_cooperative_matrix_gemm_answers_what_the_baseline_one_does`
//! overrides both the tile and the tier, which is right for what it asks --
//! two runs differing in exactly one thing -- and is precisely why it cannot
//! answer this. It would stay green on a build whose shipped tile had no
//! cooperative-matrix module at all.
//!
//! That is not a hypothetical gap; it is the gap that let all 146 coopmat
//! modules sit unreachable for the life of this crate. And the comment on
//! `shelled_at_tile` still explained the override by saying the default,
//! `(16, 32)`, "deliberately has none" -- false in both halves since the tile
//! widened to `(32, 32)`, which is a point that HAS one. A sentence about a
//! state that no longer exists, again.
//!
//! Measured: a 64-row prefill at the shipped tile resolves 2 of its 10
//! symbols above baseline, and they are the two that should be --
//! `affine_qmm_t` and `affine_qmm_t_residual` at `bm_32_bn_32`. A decode
//! resolves 0 of 9, because a matvec has no matrix and `affine_qmv_fast` has
//! no such build. Both numbers are now a test.
//!
//! Writing it took two tries and the first was the more useful. It sized its
//! prompt from `project::QMM_TILE` while the shell it built took the
//! FIXTURE's `qmm_tile`, and those are two deliberate copies -- written out
//! rather than read from the constant, so that a fixture and a projection can
//! be compared, which is how the last widening was caught. So the mutation
//! test passed: changing the constant changed nothing the test ran, and the
//! failure message named a tile that was not the one that fired. The test
//! reads the tile from the same place the shell does now. A mutation that
//! does not fail is not proof the code is right; the first thing to check is
//! whether the mutation reached the code at all.
//!
//! ## An intermittent this box cannot settle
//!
//! `the_cooperative_matrix_gemm_answers_what_the_baseline_one_does` asserts
//! the tiered and scalar runs DISAGREE, because agreeing to the last bit
//! means the cooperative-matrix module never loaded. It failed twice in the
//! full device suite and passed on the third run, and passes every time it
//! is run alone.
//!
//! It was then chased, and what came back is mostly a list of things it is
//! NOT. Ruled out by reading: this crate has no globals, so a
//! first-test-wins module cache cannot exist; `Modules::code` is a
//! deterministic lookup in a `BTreeMap` with no device in it; the tier comes
//! from device features read once at `Device` creation. Ruled out by
//! measuring: memory, with 22 GB free.
//!
//! One theory was good enough to be worth disproving. `turns::tiled` splits
//! a fire and retries, and a sub-fire that is no longer a whole number of
//! GEMM tiles takes the GEMV arm -- which would make the tiered and scalar
//! runs identical, exactly the observed failure. But the split fires only on
//! a `partial_tile` refusal, which is a specific named refusal and not an
//! arbitrary failure, so it cannot be reached by pressure. Wrong theory,
//! and cheaper to check than to believe.
//!
//! Then it stopped reproducing: four consecutive full-suite passes, two on a
//! fully idle GPU and one while a second device suite of this crate's own
//! was deliberately run against the same card. So it is not concurrency
//! inside the suite and not load as such.
//!
//! That left one correlation -- another VULKAN client, the wgpu worktree's
//! python at 2.4 GB, held the device during both failures -- and it did not
//! survive either. A third failure came later, and the two runs immediately
//! after it PASSED with that same process still resident and still holding
//! its memory. So the tally is three failures against roughly seven passes,
//! with the leading correlation present on both sides of the line.
//!
//! Which is worth stating plainly: the one hypothesis this section offered
//! is now disproved by its own follow-up, the same way the tiling theory
//! was. Nothing about the failure is understood beyond its signature.
//!
//! It stays recorded rather than explained. Guessing here would cost
//! nothing and prove nothing, which is the whole reason this file is written
//! the way it is.
//!
//! What DID change is what the next occurrence will be able to say. The test
//! asserted only that the two runs differ bitwise, so a failure meant "the
//! tiered run produced the scalar answer" and nothing more -- it could not
//! distinguish a cooperative-matrix build that ran and agreed from one that
//! was never reached at all. `Modules::resolved` answers, for a symbol and a
//! tier, WHICH tier the bytes actually came from, and `Fired::tiered` counts
//! the symbols in a fire that resolved above baseline. The test now asserts
//! directly that the scalar run resolves nothing above baseline and the
//! tiered run resolves something, BEFORE it compares any logits.
//!
//! So the tier is observed rather than inferred from a difference. If the
//! intermittent returns, the failure names its half: either the count is
//! zero, and the coopmat modules were not reached on that run, or the count
//! is positive and they ran and agreed -- which would be a numerics question
//! in `kernels-vulkan`, not a dispatch one here. Two very different bugs
//! that until now produced the same red line.
//!
//! # The one capability `driver-metal` had that this did not, and how it fell
//!
//! The sweep answered what this backend REFUSES. The other half of the
//! question is what it does not advertise, and comparing the two seams'
//! `DriverCapabilities` field by field left exactly one real difference in
//! `driver-metal`'s favour: it was ELASTIC -- `elastic_page_bytes` is a page
//! and `elastic_budget_pages` is a real number -- and this was not. (The
//! difference in the other direction is `kv_copy_domain_mask`, where this
//! backend advertises device-to-device and Metal advertises nothing.)
//!
//! The interesting part was that `Shell::resize_pool` here was not missing.
//! It preserved the pages that survived, refused by name a shrink that would
//! strand a seated conversation, left the pool intact when the machine would
//! not stage the new one, and all of that was proven on the device by
//! `a_cache_resized_under_a_conversation_does_not_change_its_answer`. It was
//! working code that production never reached, because `bootstrap` starts a
//! trim task only when both elastic numbers are non-zero and both were zero.
//!
//! The seam gave a reason for the zero, and the reason was false: "nothing
//! can be given back page-wise". A shrink here frees the old buffers and
//! takes smaller ones, so bytes do come back, at whatever granularity the
//! caller names. Right answer, wrong reason -- and the only way to tell those
//! apart is to measure.
//!
//! So it was measured, and the measurement found a REAL reason underneath the
//! false one. `Pool::resize` read every layer's whole old buffer down to host
//! memory and wrote a fresh one back up, so the charge was the pool's size
//! twice and the delta did not enter it. At 256 pages of qwen3-0.6b, handing
//! back ONE page took 2.77 s and handing back a hundred and twenty-six took
//! 0.74 s. The deeper cut was nearly four times cheaper, because the
//! destination it filled was smaller. **The cheapest trim that pool offered
//! was the largest one**, which is the opposite of what a trim task is for.
//!
//! ## Where the cost actually was
//!
//! In the ROUTE, not in the pool -- and that is only visible next to the
//! read-back finding above. Mappable VRAM is write-combined, so reading a
//! device buffer through its mapping runs at ten megabytes a second. A resize
//! did exactly that, for the whole pool, and then wrote it all back. The
//! delta-independence and the inversion were both symptoms of a cost that was
//! never about pages at all.
//!
//! `vkCmdCopyBuffer` moves the same bytes without leaving the card.
//! `Pool::resize` now takes the new buffer with `Device::empty`, copies what
//! survives device-to-device, and fills a grow's tail with
//! `vkCmdFillBuffer`. No host memory is held and nothing crosses the bus. The
//! same two shrinks take 20.5 ms and 18.6 ms, and a grow back to full takes
//! 20.0 ms: a hundred and thirty-five times cheaper, and flat rather than
//! inverted, because what is left is fifty-six allocations rather than half a
//! gigabyte of traffic.
//!
//! ## A shrink is not a grow
//!
//! The second objection was peak memory: the old resize took every new buffer
//! before freeing any old one, so it peaked at both sizes at once, and a
//! shrink asked for under memory pressure needed more memory than not
//! shrinking. That objection is right, and it is right only about SHRINKS.
//!
//! A grow keeps the all-or-nothing swap, because a pool that half-resized
//! would have some layers at the new page count and some at the old and
//! `Shape::slot` would index every one of them wrongly -- and a grow that
//! cannot get the memory must leave the pool exactly as it was. A shrink
//! migrates layer by layer instead: take the smaller buffer, move what
//! survives, free the larger one. It peaks at the old pool plus ONE layer of
//! the new, and after the first step it is monotonically decreasing, because
//! each step frees more than the next one takes. The allocation that could
//! fail is the first, before anything has moved.
//!
//! So both numbers are now published, the trim task runs, and a server pays
//! one resize to come down from its configured capacity to its working size
//! -- the target is the committed high water, which is monotonic, so every
//! later tick skips.
//!
//! ## The property that held by accident
//!
//! The old resize built each new buffer as `vec![0u8; bytes]`, so a grow's
//! new pages were zero for free. Removing the host buffer removed the zeros
//! with it, silently: Vulkan does not clear a fresh allocation, `sdpa_paged`
//! reads a whole page and lets `kv_len` decide what counts, and bf16 garbage
//! includes NaN. Hence the explicit `vkCmdFillBuffer`.
//!
//! What makes this worth writing down is that the mutation which deletes that
//! fill CANNOT BE KILLED on this card. Every test stays green, because this
//! driver zeroes fresh device memory as a process-isolation guarantee -- an
//! implementation's promise, not the specification's. `the_pages_a_grow_adds_
//! are_zero` asserts the observable property and says in its own header that
//! it cannot falsify the fill, and `zero_writes_only_the_range_it_names`
//! covers the part that IS this crate's. A green test whose subject is
//! unfalsifiable on the box it runs on is worth having and worth labelling as
//! such; the alternative is a reader who thinks it proves more than it does.
//!
//! ## The same defect, one verb over
//!
//! Fixing the resize named the real fault in words general enough to check
//! elsewhere: not "the pool was slow" but "a host `memmove` of write-combined
//! VRAM reads at about thirty megabytes a second, and this driver reaches for
//! one whenever it moves bytes it already owns". Asking where else that
//! sentence was true found `Device::copy_within`, which is what
//! `Pool::copy_page` calls once per layer per half -- and `copy_page` is what
//! every prefix share and every fork is made of.
//!
//! It had a doc comment headed "Why on the host". The reasoning in it was
//! sound and the conclusion was wrong, which is the combination worth
//! recognising: every buffer here IS host-coherent, so a `memmove` IS a copy
//! with no command buffer and nothing in flight. It answered whether the host
//! route WORKS and never asked what it costs. Measured on this model's cache
//! -- fifty-six layer-halves of thirty-two kilobytes each -- eight page moves
//! took 502.6 ms through the mapping and 27.3 ms on the copy engine, or 62.8
//! ms against 3.4 ms a page.
//!
//! The route is now chosen by whether the two ranges are DISJOINT rather than
//! by size. The crossover is about a kilobyte and a half, and the asymmetry
//! either side of it is the argument: below it the mapping saves twenty-three
//! microseconds, above it the copy engine saves thirty-three milliseconds at
//! a megabyte. The mapping is kept for the overlapping case, because a
//! `vkCmdCopyBuffer` whose regions overlap within one buffer is undefined --
//! and that is not theoretical here: the mutation that sends an overlapping
//! copy to the copy engine anyway is killed by a test comparing against a
//! host-computed `memmove`, so this card really does corrupt it.
//!
//! Both defects were invisible to every correctness test in this crate, twice,
//! for the same reason: the bytes were right. The tripwires that now stand
//! over them -- `a_trim_costs_milliseconds...` and `moving_a_page_costs_
//! milliseconds_rather_than_tens_of_them` -- are order-of-magnitude ceilings
//! rather than benchmarks, wide enough not to fire on a shared box and narrow
//! enough that restoring either host route fails them.
//!
//! ## And a third time, for bytes it never had to move at all
//!
//! The sentence kept paying. Applied a third time -- this time to bytes the
//! driver CREATES rather than moves -- it found the fire's arena, which was
//! made with `Device::buffer(&vec![0u8; n])`: a zero-filled `Vec` in system
//! memory, then uploaded whole.
//!
//! For a decode that is 326 KB and nobody would ever look. The arena is
//! sized `rows * vocab * 4`, though, so a 384-row prefill of qwen3-0.6b's
//! 151,936-entry vocabulary allocates **233 megabytes** -- and that phase
//! cost **35.5 ms of a 167 ms step, 21% of the prefill**. Through
//! `Device::empty` and a `vkCmdFillBuffer` the same arena costs **1.6 ms**.
//!
//! Nothing was misbehaving. `Device::write` ran at the 10 GB/s it documents,
//! and 233 MB at 10 GB/s is 23 ms; the host memset of the `Vec` is most of
//! the rest. The bus was asked to carry a quarter of a gigabyte of zeros to a
//! card that can write them in place at its own memory bandwidth.
//!
//! What is worth keeping is how it was found, because it was NOT found by
//! looking. `a_decode_step_does_not_stall` carried a phase table headed "and
//! still true in proportion", and remeasuring that claim is the whole story:
//! `lower` had gone from 0.8-6 ms to 0.3 MICROseconds and the pool stage from
//! 2.5-6 ms to 3, so the table's conclusion -- the fire dominates -- had
//! survived while every number under it stopped being true by three and four
//! orders of magnitude. The arena was the one phase that had not moved, which
//! on a decode makes it 3% and invisible, and on a prefill 21%. A profile
//! taken once and asserted to hold "in proportion" is a profile that can only
//! ever confirm itself.
//!
//! The tripwire is a different shape from the other two. Those had to be
//! stopwatches, because the two routes move identical bytes and only a clock
//! can tell them apart. Here there is a better witness -- whether the bytes
//! crossed the bus at all -- so `Device::uploaded` counts them and
//! `a_prefills_arena_does_not_cross_the_bus` asserts a fact about traffic:
//! 13,716 bytes uploaded against a 233 MB arena, and 125 MB when the
//! mutation puts the host route back. No ceiling to tune, and nothing for a
//! shared box to make flaky.
//!
//! And then a fourth, on the way in. `Pool::open` zeroes every layer-half of
//! the cache -- it must, since a cache holding the last model's rows would
//! produce plausible attention over sequences nobody asked about -- and it
//! made those zeros with one `vec![0u8; layer_bytes]` uploaded to each of the
//! `2 * layers` buffers. Opening a 28-layer, 512-page pool wrote **939 MB and
//! took 162 ms**; through `Device::empty` and a fill it is **0 bytes and
//! 36 ms**. That pool is small, and a serving pool is sized to fill the card.
//!
//! What makes this one instructive rather than merely repetitive is that
//! `Pool::resize`, three hundred lines further down the same file, already
//! zeroed its new tail with `Device::zero`. The right answer was already in
//! the file, next to the wrong one, doing the same job. Neither could notice
//! the other: both produce a cache full of zeros, which is the property every
//! test in this crate checks.
//!
//! ## A measurement that bought nothing, written down anyway
//!
//! With the host routes closed, the fire is 89% of a decode, and barriers are
//! most of the fire: ablating every one of them -- wrong answers, pure probe
//! -- takes a step from 8.78 ms to **2.46 ms**. So 311 barriers over 542
//! dispatches are worth about 72% of a decode, and the obvious question is
//! how many of them are real.
//!
//! One candidate looked free. `dispatch` marks EVERY bound buffer as written
//! when a row states no operands, weights included, and 56 of the table's 100
//! rows state none. `slangc` records `readonly` as `NonWritable`, and across
//! all 666 modules **3060 of 3795 bindings (80%) are read-only**, so reading
//! the mask off the module ought to have deleted a great many barriers.
//!
//! It deleted none. The count was 311 before and 311 after, and the step did
//! not move. Two reasons, and neither was visible from the code. The branch
//! is very nearly dead: a specialised name like `affine_qmm_t_bfloat16_gs_128`
//! resolves through `sig_in` to a PARENT row that does state operands, and
//! instrumenting it counted zero unstated dispatches in a decode. And the
//! parent rows' masks were already right --
//! `a_rows_writable_buffers_are_the_ones_its_module_may_write` now compares
//! all 1011 bound buffers across 189 modules against the modules' own
//! decorations and finds zero disagreements in either direction. There was
//! nothing to correct.
//!
//! The other candidate was false serialization, and it was counted rather
//! than argued. If the arena were handing independent operations overlapping
//! ranges -- and a decode has independent pairs, the three projections, the
//! gate and the up -- the false dependencies would appear as write-after-read
//! and write-after-write, since a false dependency is exactly one where no
//! value flows. Classified, the 311 are **283 read-after-write, 28
//! write-after-write, 0 write-after-read**, summing to the count exactly.
//!
//! So 91% of the serialization is one dispatch reading what the one before it
//! wrote, which is what a forward pass IS. Fewer of them means fewer kernels,
//! not better bookkeeping, and that is a `kernels-vulkan` question. Both
//! cheap explanations for the largest remaining cost in a decode are now
//! closed by measurement, which is worth more than either would have been.
//!
//! ## What the grids say, and the one they pointed at
//!
//! With both barrier explanations closed, the next question was what the
//! dispatches themselves ask the card for, so every grid a decode launches
//! was dumped. The dispatch count does not move with context -- 452 at six
//! tokens of history and 452 at seven hundred and twenty, so nothing here
//! launches per page -- and `geometry` is giving attention exactly the shape
//! it promises, heads on x and rows on y.
//!
//! That shape is the ceiling. A single-stream decode fires
//! `sdpa_paged_decode` as `[16, 1, 1]` workgroups of 128 threads: one
//! workgroup per query head, one row. Sixteen workgroups is sixteen of this
//! card's hundred and twenty-eight multiprocessors, and a workgroup cannot
//! span one, so seven eighths of the GPU is idle through the phase that is
//! two thirds of a long-context step. Nothing the driver can pass fixes that:
//! the KV range would have to be split across workgroups and merged, which is
//! a second kernel and a second dispatch, and this driver's planner is one
//! dispatch per lowered launch by construction. It is written down here as
//! the next real target rather than attempted in passing.
//!
//! ### The same claim, measured rather than reasoned
//!
//! "Two thirds of a long-context step" above was arithmetic about grids, not
//! a measurement, because until `PIE_VULKAN_TIMING` existed this crate could
//! not measure it. Every performance number here was wall-clock around a
//! `queue_submit`/`wait_for_fences` pair, which is one number for four
//! hundred and fifty dispatches; a Vulkan timestamp query pool, two writes
//! per dispatch, is what makes the per-kernel question askable at all. It is
//! opt-in, and `Device::timings` states what a slice does and does not mean.
//!
//! Pointed at a decode, the first thing it said looked like a refutation. At
//! twenty-four tokens of history the projections are **69%** of device time
//! and `sdpa_paged_decode` is **19%** -- attention a fifth, not two thirds,
//! and the target above apparently aimed at the cheap phase.
//!
//! It is not a refutation, and the reason is the whole point. At 384 tokens
//! the same measurement gives attention **75%** and the projections **22%**.
//! Both are right. The projections read the weights, which do not grow with
//! the conversation, so their cost is fixed per step; attention reads the
//! history, so its cost is linear in it. There is no single answer to "where
//! does a decode go" -- there is a fixed part and a growing part and a
//! crossover between them, and quoting either end alone is how a tuning
//! effort ends up aimed at the phase that was already cheap. The occupancy
//! work is justified, for long conversations, and would buy under a tenth of
//! a short one.
//!
//! The same run measures the host, which nobody had a number for: in
//! release a short step is 42% not-computing and a long one 15%. At short
//! context the driver's own overhead is nearly three times the whole
//! attention kernel.
//!
//! `tests/device.rs` holds both ends rather than either, and the crossover
//! between them, since the crossover is the claim neither end makes alone.
//!
//! ### The tool cost two thirds of what it was measuring
//!
//! Which was found by checking it against something else. Timing the phases
//! of `run_all` against the wall gave a submit-and-wait of 3.47 ms for a
//! short-context step -- while the timestamps were reporting 4.51 ms of
//! device work inside it, which is impossible. The difference is the tool:
//! with `PIE_VULKAN_TIMING` on, the same submit takes **5.78 ms instead of
//! 3.47**. A `BOTTOM_OF_PIPE` write after every dispatch is a point the card
//! cannot finish early, so the neighbours that had been overlapping stop.
//!
//! Nothing above changes, because everything above is a share and the shares
//! survive an effect this size -- a fifth against three quarters is not two
//! thirds of overhead. What changes is what may be said with it: the absolute
//! milliseconds are an upper bound, "this kernel takes N ms" is not a claim
//! this tool can support, and the driver's own overhead has to be measured
//! with it OFF. `Device::timings` says so where someone reaching for it will
//! read it.
//!
//! ### The host, measured with the tool off
//!
//! Release, per decode step, `run_all` broken into its phases against the
//! wall. The tool is [`phase`], turned on by `PIE_VULKAN_HOST_PHASES` and
//! driven by `tests/hostprof.rs`; the first record of this table was four
//! timers edited into a working copy, which is why it has no middle column:
//!
//! | phase | first record | before | after (short) | after (long) |
//! | --- | --- | --- | --- | --- |
//! | argument checks | 0.007 ms | -- | 0.007 ms | 0.006 ms |
//! | descriptor sets | 0.134 ms | -- | 0.123 ms | 0.121 ms |
//! | command recording | 0.421 ms | -- | 0.383 ms | 0.383 ms |
//! | submit and wait | 3.469 ms | 3.437 ms | 2.870 ms | 12.070 ms |
//! | outside `run_all` | 1.917 ms | 4.94 ms | 1.44 ms | 1.39 ms |
//! | wall | 5.949 ms | 8.377 ms | 4.803 ms | 14.150 ms |
//!
//! The host was 2.48 ms when this was first written and **it does not move
//! with context**: the same milliseconds at 24 tokens and at 384. Three
//! quarters of it is outside `run_all` altogether -- the lowering, the plan,
//! the scalar blocks -- which was a part of this driver no measurement here
//! had ever looked at. The recording of four hundred and fifty dispatches is
//! 0.38 ms of it and the descriptor sets 0.12, both of which had been assumed
//! to be the expensive part and are not.
//!
//! Looking properly found that it had grown rather than held: the crossing to
//! `hold::arm_for` had put a discarded SPIR-V parse into `serve::fire` once
//! per RECTANGLE rather than once per symbol, and by the time a repeatable
//! tool existed that was 3.18 ms of a 4.94 ms host step. Caching it, the arm
//! lookup, the pipeline probe and the arena buffer -- all four of them things
//! a decode recomputes identically every token -- took a short step from 8.38
//! ms to 4.80 and a long one from 18.02 to 14.15. See
//! `the_projections_dominate_both_steps_now_that_the_decode_splits_its_keys`
//! for the phase-by-phase table.
//!
//! So there are two targets and they belong to different regimes. Long
//! conversations are attention, and the occupancy work above is the answer.
//! Short ones are the host, and what is left of it -- 1.4 ms, almost all of
//! it planning 452 rectangles at 1.5 microseconds each -- comes down by
//! stating fewer rectangles rather than by caching anything more.
//!
//! # BOTH OF THOSE CONCLUSIONS WERE READ OFF A BROKEN SUBTRACTION
//!
//! The `outside run_all` row is `median - run_all`, and after `replay`
//! landed, ninety-four percent of steps never enter `run_all` -- they submit
//! an already-recorded command buffer and wait on a fence. So that row is not
//! host time, it is host time PLUS the entire fence wait, and the closer this
//! driver got to replaying everything the more of the card's own execution it
//! swallowed. It reported 1.4 ms of host. Subtracting the
//! `fire/replay/submit` span as well, the host is **0.09 ms** -- six percent
//! of a step, not thirty.
//!
//! The row is left standing because the columns beside it are a historical
//! record and the SPIR-V regression it caught was real; what is retracted is
//! reading it as a host figure. The lesson is not that a number was wrong.
//! It is that this one was wrong in the flattering direction: a profile that
//! says the host owns the step is a profile that sends the next week's work
//! to the host, and it will keep saying so no matter how much of the host you
//! delete, because what it is really measuring is the GPU.
//!
//! Both targets are on the CARD, and they are the same target. The step is
//! 1.62 ms, of which 1.51 is inside the submit; and of THAT, 1.06 ms is not
//! arithmetic but ORDERING -- delete the pipeline barriers and the same
//! dispatches, giving wrong answers, finish in 0.46 ms. 452 dispatches with
//! 311 barriers between them, each dispatch a single row of a 0.6b model and
//! therefore almost entirely launch latency, and the barriers stop the card
//! from hiding one behind another. See `hazards` in `device.rs`.
//!
//! "State fewer rectangles" survives as the conclusion. Every reason given
//! for it here was the wrong reason.
//!
//! The dump did find something it was not looking for, one crate over.
//! `sdpa_vector.slang` -- the dense decode, off this model's path -- still gave
//! every thread its own copy of the whole dot product, which is the exact
//! defect `sdpa_paged.slang` was cured of when 1536 tokens went from 306 ms to
//! 110 ms. So did `sdpa_sliding.slang`, and so did `sdpa_paged_mma.slang`. Four
//! bodies descend from one Metal template and share an online softmax but not
//! a dot product, so fixing one could not fix the others and nothing pointed
//! from any of them to its siblings; the cure had been applied to exactly the
//! one that had been profiled. Transcribing the tree reduction across is
//! worth about 1.5x on the dense key loop and 2.0x on the tiled one, both
//! measured; the shaders say why those are not the 128x and 64x their
//! arithmetic counts imply.
//!
//! The fourth was the one that made the search worth running rather than
//! merely tidy. Its redundancy was doubled -- thirty-two lanes shared a row
//! AND each re-ran the whole tile loop per output slot -- and its fix is not
//! the other three's, because its accumulation sits under a `mine` guard and
//! a barrier inside a guard only some threads pass is undefined. Finding it
//! also turned up the gap that had hidden it: the kernel's only test ran
//! sequences of five tokens against a sixteen-token tile, so the loop that
//! carries an online softmax from one tile to the next had never run twice.
//! A body that reset its running maximum every tile passed that test. It does
//! not pass the one added beside it.
//!
//! ## The defect only the whole sweep could see
//!
//! Re-running the curated inferlet sweep afterwards -- expected to be a
//! formality, since none of those four kernels is on the qwen3 path -- read
//! 34 of 39 rather than 35. The new failure was `prefix-tree-kv-cache`, and
//! it passed when run alone. It fails after the other thirty-eight because
//! this pool is ELASTIC: it holds what the frames so far have needed, not
//! what the scheduler is entitled to hand out, and it grows in `Shell::admit`
//! to the highest page a frame NAMES. A copy plan is the OTHER door a page
//! number arrives through, and it did not carry that reasoning -- so a prefix
//! share aimed one page past the last prefill's high-water mark was refused
//! by a bounds check that was right about the pool as it is and had no way to
//! know what it could be.
//!
//! The fix is four lines and the reasoning is one asymmetry: grow for
//! DESTINATIONS, keep refusing SOURCES. A page this pool has never held is a
//! page nothing has ever written, so growing for a source would turn a
//! refusal into a copy of fresh zeros -- history-shaped silence instead of an
//! error.
//!
//! Two things are worth keeping from it. The elastic pool was tested, and
//! every one of those tests was about `resize_pool` and `admit`, because
//! those are where the elasticity was WRITTEN; `copy_kv` was tested too, and
//! every one of those tests was about arithmetic on a pool big enough for it.
//! Neither suite could ask the question the other one answers. And a driver
//! whose answer depends on which requests preceded it is invisible to any
//! per-test harness -- the sweep found this only because it runs thirty-nine
//! programs against one server.
//!
//! Asking the same question of the OTHER door found the same answer. A
//! frame's growth came from `kv_translation` and `required_kv_pages`, which
//! are the engine's statements ABOUT its pages, and not from
//! `kv_page_indices`, which is the list this driver binds. Those can differ:
//! the translation is empty whenever nothing moved, and only one of the
//! engine's two batch-assembly paths folds the page list into the declared
//! high-water. A frame that named page 7 on a three-page pool died with
//! `Unstageable(NoSuchPage { page: 7, pages: 3 })` -- measured, not reasoned
//! about, by firing one.
//!
//! That one was not silent: `Request::stage`'s bounds check caught it, which
//! is exactly why that check stays where it is. It was a request KILLED for a
//! page the pool could have grown to hold. Sizing the pool by what will be
//! bound rather than by what was declared is strictly more permissive than
//! trusting the declarations and never less, because the declarations are
//! still maxed in.
//!
//! Both doors are the same lesson, and it is worth stating once: a driver
//! with an elastic pool has to grow wherever a page number ENTERS, and there
//! were three entrances with the growth written at one of them.
//!
//! What makes that claim checkable rather than argued is the pool's opening
//! size. It opens at 1024 pages and no curated inferlet fills it, so a
//! default run barely enters the growth path at all -- which is why both
//! defects survived a suite that exercises everything else. `[model]
//! kv_pages` was supposed to be the knob for that and was rejected by the
//! worker's own config before either driver that reads it saw the bytes;
//! with that fixed, the measurement is:
//!
//! ```text
//!   1024 pages, before   34/39
//!      8 pages, before   34/39
//!      8 pages, after    35/39
//!      1 page,  after    35/39
//! ```
//!
//! The whole sweep runs from a pool opened at ONE page. Every page any of
//! those thirty-nine programs touches is one this driver grew for, through
//! all three entrances, while the programs were running.
//!
//! ## And then the sweep was made to run more than one thing at a time
//!
//! Every case above ran alone. A sweep that never overlaps two requests is
//! green about the half of this driver that is easy: the batching, the
//! paging, and the pool growth are all machinery for serving many at once,
//! and nothing was asking them to. The question is cheap to ask -- the same
//! prompt at temperature 0, run alone, then 2, 4 and 8 ways at once, has to
//! give back the same tokens -- and anything one request can see of another,
//! a page or a row of scores or a slot in a batch, comes back as a lane that
//! diverged.
//!
//! Nothing diverged, at any width. That is the strongest evidence so far
//! that the six cross-request reads closed earlier in this file are actually
//! closed, and it is now `greedy-decoding-is-the-same-alone-and-in-a-crowd`
//! in the curated sweep rather than a thing someone once ran.
//!
//! The defect it found was a layer up and is recorded here because of how it
//! hid: eight launches install one program at once, and the server keyed
//! in-flight uploads by the program's HASH, so all eight were one entry in
//! one map. Sequential callers never collide, so the bug was invisible for
//! exactly as long as the tests were polite.
//!
//! The parsing survives the negative result because the test needs it, and
//! the test is worth having for a reason the optimization never had: a row
//! that understates a write does not make this driver slow, it makes it
//! race, and a race this card usually wins is the one defect a passing test
//! suite cannot be trusted about.
//!
//! The same field-by-field pass turned up one more constant with no check
//! behind it: `rs_cache_required: false`, where `driver-metal` answers
//! `deployment.recurrent.is_some()`. Six comments in this crate say "no
//! model this driver serves holds a recurrent state", and the catalog has
//! three families that project one -- `nemotron_h`, `kimi_k3`, `qwen_3_5` --
//! with nothing visibly routing them away.
//!
//! A guard was written for the seam and then measured, and it never fired.
//! Every one of those rows is already refused a line earlier by `row.trace`,
//! for having no Metal text, in sentences better than any this crate would
//! write: qwen-3.5 says its forward "interleaves gated DeltaNet layers with
//! attention", nemotron-h says the one Metal text here "has no recurrent
//! layer kind", and both name the backend that does serve them. So the guard
//! was deleted and a test kept in its place. The belief is true; this crate
//! is simply not what makes it true, and a guard that cannot fire is a claim
//! that cannot be checked.
//!
//! # How fast it actually is, which nobody had asked
//!
//! Every number in this crate until now was taken under two validation
//! layers, where absolute cost means nothing. So a decode was measured
//! properly: release build, no layers, a 4090, qwen3-0.6b at 4 bits, one
//! conversation. **44 ms per token, about 22 tokens a second.**
//!
//! Both that number and the 18.1 ms it later became are SUPERSEDED, and the
//! section "Ninety-eight per cent of a prefill was one uncached memcpy" below
//! says by what. **Where it stands today: 5.4 ms a decode step at 24 tokens
//! of context and 15.0 ms at 384, release, same box, same model** -- about
//! 185 tokens a second, against the 22 this section opens with. Nothing in
//! this section was the cause of that; each fix below names its own share.
//! The current figures are repeated here because a narrative that only ever
//! supersedes numbers leaves a reader to guess where it landed, and the two
//! tripwires that guard them -- `a_decode_step_does_not_stall` and its long
//! sibling -- carried figures three to eight times stale for months for
//! exactly that reason.
//!
//! The accounting in this section is left as it was written
//! because its conclusion -- "the driver is not the problem" -- turned out to
//! be the thing that was wrong, and it was wrong for a reason worth keeping:
//! it timed the lowering, the descriptor writes and the recording, and it
//! never timed reading the answer back.
//!
//! That is slow, and worth knowing precisely rather than vaguely. Of the
//! 44 ms, 38 are the GPU: this crate's own bookkeeping -- lowering,
//! descriptor writes for 452 launches, recording the command buffer -- is
//! under a millisecond together, and the arena it allocates per step is
//! 326 KB and costs 0.2 ms. The remaining host cost is the state uploads and
//! the lowering, at a few milliseconds. So the driver is not the problem, and
//! now there is a number saying so instead of a hope.
//!
//! The device half is a `kernels-vulkan` question. 38 ms over 452 serialized
//! dispatches is 84 us each; `affine_qmv_fast` accounts for about 27 ms and
//! `sdpa_paged_decode` for about 16 ms; the lm_head launch reads its 78 MB of
//! weights at roughly 12 GB/s on a card that does about a thousand, and the
//! card draws 85 W of 450 at "100% utilisation" -- resident and stalled.
//!
//! Three plausible causes were tested and are NOT it, which is the part worth
//! recording because each cost an experiment:
//!
//! 1. The quantised matvec re-fetches each packed word once per code -- eight
//!    times at 4 bits -- and its scale and bias once per element. Rewriting
//!    the inner loop around `pie_affine_word_dot` to load each word once
//!    moved the step from 44.4 ms to 44.3 ms; the shader compiler was already
//!    hoisting them. Reverted: a change with no measured benefit that
//!    perturbs rounding is not an improvement.
//! 2. Its reduction has one lane of 64 sum 32 shared floats per row. Deleting
//!    the reduction entirely -- wrong answers, pure probe -- gave 45.1 ms.
//! 3. Every dispatch is separated by a global barrier, so none overlap.
//!    Removing every barrier gave 39.6 ms: about 5 ms of 44, real but not the
//!    story.
//!
//! What remains is the kernels' decomposition, which is a larger piece of
//! work than a note. `a_decode_step_does_not_stall` holds a tripwire at
//! 400 ms -- an order of magnitude, not a benchmark -- because the one
//! performance defect this driver has actually shipped, a 370 s KV copy, was
//! found by running something rather than by any test, while every
//! correctness gate stayed green. Answers can be right while the clock is
//! wrong, and nothing here was watching the clock until now.
//!
//! # Twenty-four tokens is not a conversation
//!
//! Everything above was measured from a twenty-four token prompt, and that
//! turned out to be the more interesting mistake. Decode cost has two parts:
//! a fixed one, twenty-eight layers of weights read whatever the history, and
//! one that GROWS with the history, attention reading every key it has kept.
//! At twenty-four tokens the second is invisible. Measured across three
//! contexts it was not:
//!
//! | context | decode step | after the fix below |
//! |---|---|---|
//! | 24 | 38.0 ms | 31.8 ms |
//! | 384 | 100.6 ms | 49.9 ms |
//! | 1536 | 306.4 ms | 110.1 ms |
//!
//! 0.177 ms per token of history. At an ordinary thousand-token conversation
//! attention was five sixths of every step, so the profile above -- and the
//! three refuted hypotheses, all of which are about the matvec -- had been
//! written about the part of the step that barely matters.
//!
//! The cause was structural and plain once looked at. A decode workgroup in
//! `attn/sdpa_paged.slang` runs one thread per head dimension, and every one
//! of those threads was walking the whole query and key vectors to arrive at
//! the same scalar score: a hundred and twenty-eight threads computing one
//! number a hundred and twenty-eight times. Having the workgroup contribute
//! one term each and add them in a tree is 0.052 ms per token, and 2.8x
//! faster at 1536. `a_long_conversations_decode_step_does_not_stall` is what
//! holds it, because none of the fifty-odd device tests before it ran a
//! context long enough to see this at all.
//!
//! Asking for the same thing at 1536 tokens under the validation layers found
//! a second defect, in this crate rather than in a kernel. A prefill tile
//! that large does not finish inside `run_all`'s ten-second fence wait, which
//! was reported correctly -- and then buried, because `serve::fire` gives the
//! scalar block back on the failing path too, and freeing a buffer that the
//! just-recorded descriptor sets still name is a validation error this driver
//! treats as fatal. So the process aborted on the consequence and never
//! printed the cause. `run_all` now waits for the device to go idle before a
//! failed fire returns, and frees the fire's descriptor sets at the end of
//! the fire that used them rather than at the start of the next one.
//!
//! # The bus, which is what all of it was
//!
//! Four hypotheses about the shaders were tested and refuted -- the matvec's
//! redundant word loads, its serial reduction, the barrier between every
//! dispatch, and its occupancy at 32 lanes to a row. Each was plausible, each
//! was measured, and each changed nothing. What they had in common is the
//! thing that should have been read first: every kernel ran at about 12 GB/s
//! REGARDLESS of size -- the lm_head launch streaming 78 MB and a small
//! projection touching 1.5 MB alike -- on a card whose memory does roughly a
//! thousand. That is not the signature of a decomposition. It is the
//! signature of a bus.
//!
//! `Device::buffer` asked for the first memory type that was
//! `HOST_VISIBLE | HOST_COHERENT`, because every buffer here is written
//! through a mapping and this driver has no staging path. That requirement is
//! real; taking the FIRST match was the mistake. This card offers five types,
//! and the first host-visible one is system RAM:
//!
//! | type | flags | heap |
//! |---|---|---|
//! | 1 | `DEVICE_LOCAL` | 24 GB, VRAM |
//! | 2 | `HOST_VISIBLE \| HOST_COHERENT` | 47 GB, system RAM |
//! | 3 | + `HOST_CACHED` | 47 GB, system RAM |
//! | 4 | `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT` | 24 GB, VRAM |
//!
//! So every weight, every KV page and every activation lived in system memory,
//! and each of the 452 dispatches in a decode step reached across PCIe for all
//! of it. Type 4 is the same memory the card computes out of, mappable across
//! its whole twenty-four gigabytes because resizable BAR is on, and it costs
//! nothing to prefer. A decode step went from 31.8 ms to 18.1 ms, and one at
//! 1536 tokens of history from 110 ms to 59 ms. With the attention fix above,
//! 5.2x on a real conversation.
//!
//! Preferred and not required, in both directions: a part with no device-local
//! host-visible type gets what it always got, and an allocation that fails on
//! the smaller VRAM heap falls back to system memory, so a model that no
//! longer fits serves slowly instead of being refused. That fallback is also
//! why `Device::budget` still answers with the largest host-visible heap: it
//! is a bound on what can EVER be allocated, and that is still true.
//!
//! The price is paid by the tests rather than by serving. Mapped VRAM is
//! write-combined, so host READS of it are slow, and the device suite went
//! from 334 s to 483 s because so much of it reads buffers back to compare
//! them. The twelve GPU gates, which read back only what a client asked for,
//! got faster: boot 2.3 s to 1.6 s, the shared-prefix gate 106 s to 79 s.
//!
//! None of this says the kernels are good. It says they were not what was
//! wrong, which is a different and much cheaper thing to know.
//!
//! # Prefill, which is a different shape of slow
//!
//! Everything above is about decode. Prefill was measured next, one fire per
//! prompt, release build, layers off:
//!
//! | prompt | before | after |
//! |---|---|---|
//! | 192 tokens | 2.65 s | 1.76 s |
//! | 768 tokens | 25.75 s | 23.41 s |
//! | 1536 tokens | 54.47 s | 49.61 s |
//!
//! "After" is `sdpa_paged.slang`'s tiled path made cooperative the same way
//! the decode path already was -- thirty-two lanes reducing one score between
//! them instead of each lane recomputing the whole dot product for every
//! dimension it owns. It is a third off a short prompt and a tenth off a long
//! one, and the shape of that is the interesting part: the win SHRINKS as the
//! prompt grows, which is the opposite of what a quadratic term does. So
//! attention is not what prefill spends its time on.
//!
//! What it spends it on is the GEMM, and the number that says so is the one
//! per token: **32 ms a prompt token at 1536, against 37 ms for a decode
//! STEP**. A decode step is one row through the whole model. A prefill row
//! costs almost the same as one, which means the batched GEMM is buying
//! almost nothing over doing the rows one at a time.
//!
//! The reason turned out to be TWO things, and the first one was mine.
//!
//! ## The tier that was never running
//!
//! `kernels-vulkan` compiles a tiered module beside each baseline one and
//! names it `<entrypoint>.<tag>.spv`. Every module store here -- the engine's
//! `read_modules`, the test fixtures -- is keyed by FILE STEM, so the
//! cooperative-matrix build of `affine_qmm_t_..._bm_32_bn_32` sits under the
//! key `affine_qmm_t_..._bm_32_bn_32.coopmat`.
//!
//! A plan never names that. A plan states the bare entrypoint, because the
//! tier is a property of the DEVICE and not of the text. And `Modules::code`
//! took only a symbol. So the lookup could not reach a tiered module even in
//! principle: **all 146 cooperative-matrix modules and all 20 fp16 ones were
//! dead**, on every device, in production and in tests alike, from the first
//! commit of this crate.
//!
//! Nothing failed. `Device::tiers()` reported `Coopmat` first, `Shell` set
//! `tier: Coopmat`, the pipeline cache keyed on it, `serve::fire` carried it
//! and the seam advertised it. Every part of the machinery agreed the tier
//! was in use except the one line that had to name the file. `module_for`,
//! the tier-aware resolver on `Device` that does this correctly, was called
//! only by tests.
//!
//! It is invisible by construction -- a tier that is off looks exactly like
//! a tier that is on but does not help -- so it was found by measuring
//! instead. Prefill at 1536 tokens cost 56.1 s at a GEMM row tile of 16,
//! 54.7 s at 32 and 54.4 s at 64, when 32 and 64 have a cooperative-matrix
//! module and 16 deliberately does not. **A tier that changes nothing when
//! you switch to it is a tier that is not running.**
//!
//! `Modules::code` now takes the tier and walks `Capability::PREFERENCE`
//! down from it, so a device at the top tier still gets the baseline module
//! for the great majority of entrypoints that have only one. Three tests in
//! `serve.rs` pin all three directions, including that a baseline device
//! never reaches a coopmat file -- handing it one is a module declaring a
//! capability the device did not enable, which is a validation error rather
//! than a slow answer.
//!
//! ## What that is worth, once the tile can use it
//!
//! With the tier reachable, the same three-tile measurement:
//!
//! | GEMM row tile | before | after |
//! |---|---|---|
//! | 16 (has no coopmat module) | 56.1 s | 56.1 s |
//! | 32 | 54.7 s | 39.4 s |
//! | 64 | 54.4 s | **6.8 s** |
//!
//! **Eight times.** Neither change does anything alone, which is exactly why
//! this took four wrong answers to find: the tier fix is worth nothing at
//! the tile actually in use, and the tile is worth nothing while the tier
//! cannot be loaded.
//!
//! The second half has since been taken: `QMM_TILE` is `(32, 32)`, and the
//! measurement that decided it is at the constant. It was held back on the
//! belief that a per-backend tile meant touching every `MetalBinding`
//! constructor -- which was wrong twice over. `qmm_tile` is already a field
//! rather than a read of the constant at the point of use, and all three
//! kernel trees that consume it declare the same `TILE_M` rungs, so 32 is not
//! a Vulkan-only point. See "The tile, once anything could see it" below.
//!
//! ## How much of the tiered build anything can reach
//!
//! The fix made the walk work. It did not make the tree worth walking, and
//! the audit is worth stating plainly because the previous section's eight is
//! easy to misread as something this crate shipped.
//!
//! Of **185 tiered modules -- 146 `coopmat` and 39 `fp16` -- 52 belong to an
//! entrypoint any text in this repository names.** `model_compiler`'s
//! `dsl::metal` emits exactly two quantised-GEMM stems, `affine_qmm_t` and
//! `affine_qmm_t_residual`. Everything else the tier stamps -- `_bias`,
//! `_splitk`, `_strided`, the seven `*_fp16_precast*` families,
//! `sdpa_paged_mma` and its sink twin -- is compiled for a symbol no plan
//! states. That accounts for ALL 39 fp16 modules, so **the fp16 tier is
//! unreachable by NAMING**, which is not something a resolver change can
//! lift and is not a defect in the same sense: those are distinct
//! entrypoints a text would have to ask for deliberately.
//!
//! And of the 52, **zero were reachable at the tile the constant then
//! stated**, because `QMM_TILE` was `(16, 32)` and no cooperative-matrix
//! build exists at a row tile of 16. At `(32, 32)` **twelve** are: one per
//! `(group x bits)` point of the two stems.
//!
//! So: the resolver fix unblocked an eight-fold prefill win and, by itself,
//! took none of it. Both halves were needed and the second lived in
//! `crates/model`. `the_tiered_builds_this_driver_can_actually_reach` pins
//! all of these numbers, so the day any of them moves is a day to re-read
//! this -- it is the test that went from asserting zero to asserting twelve
//! when the tile widened, which is exactly what it was written to do.
//!
//! ## Ninety-eight per cent of a prefill was one uncached memcpy
//!
//! `Device::buffer` prefers the memory type that is both `DEVICE_LOCAL` and
//! `HOST_VISIBLE`, which on this card is the whole 24 GB of VRAM behind
//! resizable BAR. That was worth five times the decode rate and is not in
//! question. What went unexamined for as long as this crate has existed is
//! the OTHER direction: mappable VRAM is write-combined, so writes through
//! the mapping coalesce and reads through it are uncached, unprefetched, and
//! one PCIe round trip deep.
//!
//! Every answer this driver has ever returned came back through such a read,
//! and `serve::logits` read the WHOLE arena to slice the logits out of it.
//! Timed by phase, on a 1024-token prefill of the real 4-bit qwen3-0.6b:
//!
//! | phase | before | after |
//! |---|---|---|
//! | allocate and zero the 334 MB arena | 82 ms | 82 ms |
//! | every dispatch of every layer | 588 ms | 588 ms |
//! | **read the answer back** | **32 967 ms** | **220 ms** |
//! | widen 155 M bf16 logits to f32 | 278 ms | 278 ms |
//! | **the whole step** | **33 847 ms** | **1 107 ms** |
//!
//! Ten megabytes a second, on a bus that does twelve gigabytes. The
//! dispatches -- the part a driver is for -- were one sixtieth of the step.
//!
//! The fix is `Device::read_at`: a staging buffer in host-cached system
//! memory, one `vkCmdCopyBuffer`, one fence, and a cached `memcpy` out of it.
//! The copy engine reads VRAM at the bus's rate and the host reads system RAM
//! at the cache's. It also reads only the readout's range rather than the
//! whole arena. Both fallbacks go back to the mapping: a buffer that is not
//! device-local is already in system memory, and a staging path that cannot
//! allocate should be slow rather than fatal.
//!
//! `a_read_of_device_memory_goes_through_the_copy_engine` pins it, and the
//! third of its three assertions is the one that would otherwise go quiet:
//! both paths answer correctly, so nothing else in the suite would notice the
//! day this stopped staging -- except the wall clock, which no test watches.
//! `Device::staged()` counts, and the test asserts the count moved.
//!
//! ## And what that did to every timing this crate had taken
//!
//! It invalidated them, decode included. The 18.1 ms step recorded above was
//! remeasured with the staged path forced off and read **18.2 ms** -- the
//! same number, on the same box, minutes before the same benchmark with
//! staging on read **11.1 ms**. A decode's arena is only 326 KB, but 326 KB
//! of uncached write-combined memory is about seven milliseconds, and seven
//! of eighteen was hiding inside a step whose profile above says the host
//! bookkeeping is "under a millisecond". It was: the readback is not
//! bookkeeping, and nothing had timed it.
//!
//! | qwen3-0.6b at 4 bits, release, no layers, one conversation | mapped | staged |
//! |---|---|---|
//! | decode, best of 64 | 10.9 ms | **8.6 ms** |
//! | decode, mean of 64 | 18.2 ms | **11.1 ms** |
//! | 1024-token prefill | 33.8 s | **1.11 s** |
//!
//! The staging buffer is kept between reads, grown to the largest one asked
//! for and never shrunk -- the rule `Scratch`'s descriptor pool follows, for
//! the same reason: a server's reads are the same two or three sizes forever.
//! Allocating one per step cost about a millisecond of a decode's mean, which
//! is what a `vkCreateBuffer` plus a `vkAllocateMemory` plus a free costs on
//! this card. Decode's mean over 64 steps went 11.1 ms to **9.5 ms** and its
//! best 8.6 to 7.1.
//!
//! Where a decode's time goes now, timed by phase inside `Serving::once`
//! over eight steady-state steps: **6.25 ms firing 452 dispatches, 0.65 ms
//! reading the answer back, 0.13 ms allocating and zeroing the 326 KB
//! arena**, and about 1.5 ms above `once` in the lowering and the state
//! uploads. So the readback has gone from a third of a decode to a
//! fourteenth, and the section above is right again for the first time: what
//! is left is the kernels', not the driver's.
//!
//! The GEMV-versus-GEMM table that used to be in this section was the same
//! mistake at a larger scale -- a measurement whose readback is nine tenths
//! of it is a measurement of the readback. Retaken, minimum of seven runs
//! each, through `Shell::step` on the real 4-bit qwen3-0.6b --
//! `TokensMultipleOf(tile)` means a prompt the tile divides takes the tiled
//! GEMM and one token fewer does not, so the two arms compare directly at
//! almost identical work:
//!
//! | rows | GEMV | GEMM, tile 16 | tile 32 | tile 64 |
//! |---|---|---|---|---|
//! | 15/16 | 20 ms | 107 ms | | |
//! | 31/32 | 34 ms | | 52 ms | |
//! | 63/64 | 62 ms | | | 90 ms |
//! | 127/128 | 146 ms | 350 ms | 111 ms | 126 ms |
//! | 511/512 | **632 ms** | **1449 ms** | **420 ms** | **416 ms** |
//!
//! The shape the contaminated table could not show: at 512 rows the tile that
//! SHIPS is **2.3x slower than having no GEMM at all**, and a tile of 32 or
//! 64 is 1.5x faster than having none. `(16, 32)` is the one setting that
//! gets neither -- it pays for a GEMM and has no cooperative-matrix build to
//! spend it on, because nothing generates a 16-row coopmat variant.
//!
//! This is a measured argument for `QMM_TILE = (32, 32)` and it is not taken
//! here, for the reason `project.rs` gives: the tile is one constant shared
//! by every backend, and a per-backend one means a field on
//! `model::catalog::MetalBinding` and a touch of every constructor of it. The
//! numbers are recorded at `QMM_TILE` so that whoever adds that field does
//! not have to take them again.
//!
//! Minima and not means, because this box is shared: a first pass at 1024
//! rows once read 14.6 s and a repeat of the same size read 21.8 s while a
//! neighbouring process had the card.
//!
//! ## And the fastest read of a prefill's answer is the one not taken
//!
//! The copy engine made the readback affordable. It did not make it small:
//! the section above still reads EVERY row a fire computed, and a prefill
//! computes one per token because the lowering is told every row samples.
//! A 1024-token prompt therefore hands back 1024 x 151,936 bf16 values --
//! 311 MB off the card, widened into a 622 MB `Vec<f32>` -- and the turn that
//! asked for it wants one row.
//!
//! So [`serve::logits_of`] takes the rows a caller will name and reads the
//! one contiguous span that covers them, widening only those. `Logits` now
//! carries a `read: Vec<usize>` saying which fire rows `values` holds, and
//! `Logits::row` binary-searches it rather than multiplying. Measured on the
//! real 4-bit qwen3-0.6B, timing the read phase of `Serving::once` alone:
//!
//! | prompt | rows read | read phase, dense | read phase, narrowed |
//! |---|---|---|---|
//! | 512 tokens | 1 of 512 | 136-208 ms | **0.2 ms** |
//! | 1024 tokens | 1 of 1024 | 261-435 ms | **0.2 ms** |
//!
//! and on the whole step, best of five: 512 tokens 1432 -> 1266 ms, 1024
//! tokens 2898 -> 2639 ms. Decode is unchanged, because a decode's fire is
//! one row and it was already reading all of it.
//!
//! What makes this safe is an ordering fact rather than an argument about
//! sampling. `Serving::over` recomputes `readout_of` from the requests' own
//! order, and `Frame::of` builds `request_of_token` by walking requests in
//! that same order -- so the rows `over` names are exactly the rows `once`
//! read. In a SPLIT fire (see the partial-tile path) each half names the last
//! row of each request it contains, a request's last row lies in the half
//! that contains it, and `turns::join` renumbers the second half's rows into
//! the whole fire's numbering while dropping the overlap. Every row anything
//! downstream can address was read by the sub-fire that produced it.
//!
//! Pinned by `a_deployment_fires_step_after_step_and_stops_building_pipelines`,
//! which requires both that `logits.read == readout_of` and that `values` is
//! exactly one row wide after a sixteen-row prefill -- two facts, because
//! either alone is satisfiable wrongly. Restoring the dense read fails it.
//!
//! ## The tile, once anything could see it
//!
//! With the readback out of the way a prefill could finally be phase-profiled
//! honestly, and the answer was that it is almost entirely dispatches: of a
//! 2799 ms step at 1024 tokens, **2694 ms was `fire`**, 95 ms was zeroing and
//! uploading the arena, 2 ms was lowering, and 0.2 ms was the read.
//!
//! And nearly all of that 2694 ms was one bad constant. `QMM_TILE` was
//! `(16, 32)`; the same prompt one token SHORTER, which the
//! `TokensMultipleOf(tile)` guard sends down the matrix-vector arm instead,
//! took **1054 ms**. The tiled GEMM this driver was firing was 2.6x slower
//! than firing no GEMM at all. Overriding the tile through
//! `shelled_at_tile`, best of three after a warm-up, same 1024-token prompt:
//!
//! | tile | prefill |
//! |---|---|
//! | `(16, 32)`, as it shipped | 2563 ms |
//! | `(32, 32)` | **565 ms** |
//! | `(64, 32)` | 580 ms |
//!
//! **4.5x**, and it is now `(32, 32)`. The reason it survived so long is the
//! section above this one: every prefill number anyone had taken included a
//! readback of every row the fire computed through an uncached mapping of
//! write-combined VRAM, and that memcpy was most of the wall clock. The
//! kernel was never what was being timed.
//!
//! Two things moved with it in this crate's tests, both worth knowing about
//! because both are the shape of thing that would otherwise go quiet:
//!
//! * The two tests that fire "a whole tile" to make the prefill plan state a
//!   GEMM were transcribing sixteen. At tile 32 they were firing the GEMV arm
//!   and comparing it against itself -- passing, vacuously. They read
//!   `QMM_TILE.0` now.
//! * `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`
//!   counts workgroups over every real plan, and the count fell by 208,400. A
//!   wider tile is fewer workgroups over the same output, so that is the
//!   arithmetic working rather than work going missing.
//!
//! ## And a decode was spending a seventh of itself re-answering a question
//!
//! Phase-profiled the same way, a decode of the real 4-bit qwen3-0.6B was
//! 8.9 ms: `fire` 6.6, **`lower` 1.38**, `Pool::stage` 1.30, the state tables
//! 0.26, the arena 0.18, the read 0.21, the free 0.10. The second-largest
//! item was pure host work being redone every step for an answer that had not
//! changed.
//!
//! It could not have changed. `model_compiler::lower::lower` is a function of
//! a plan, a slice of `Row`, and a `Fire` — and a `Row` is six booleans and
//! an optional depth. No position, no history length, no page, no token.
//! Everything that differs between two decodes of one conversation reaches
//! the GPU through `Pool::stage` and the state tables instead. So
//! `turns::Lowerings` keeps the last few lowerings by the row shape that
//! produced them, keyed also on which of the two plans was used, and a
//! decode mean fell **8.94 -> 8.12 ms**.
//!
//! It is pinned by a COUNT rather than a duration:
//! `Lowerings::lowered()` says how many times `lower` actually ran, and
//! `a_deployment_fires_step_after_step_and_stops_building_pipelines` requires
//! four over its four distinct row shapes. Disabling the lookup makes it
//! eleven. A wall-clock assertion could not have made this claim -- a cache
//! that never hits returns exactly the same answers, and on a shared box a
//! duration measures the neighbours.
//!
//! ## And the tables it stages were nine allocations a step
//!
//! `Pool::stage` was next, at 1.30 ms. It writes eleven small tables -- rows,
//! positions, the paged-KV CSR, the write descriptors, the masks, the token
//! ids, the sampling identity -- and `Pool::state` allocated a fresh device
//! buffer for each and freed the old one. Sixty-six `vkAllocateMemory` calls
//! and sixty-six frees over six decodes, for a few hundred bytes.
//!
//! A conversation decoding states the same row count and the same page count
//! for eight tokens at a stretch, so those tables are the same SIZE step
//! after step. `Pool::state` now writes over the buffer already there when
//! the size matches EXACTLY, and allocates otherwise. Exactly, and not "big
//! enough": a larger buffer is bound `whole` with the previous fire's numbers
//! in its tail, and `Device::read` of a table would then answer with that
//! tail -- so a test reading a table back would be reading a different object
//! than the one the fire was given. It is safe against the GPU because
//! `Serving::once` waits on the fire's fence before returning, which the
//! free it replaces already relied on.
//!
//! Decode mean **8.12 -> 6.45 ms**, best 7.64 -> 5.43. More than the 1.30 ms
//! that was being spent inside `stage`, which is the allocator's pressure on
//! everything else.
//!
//! Pinned the same way as the lowering cache and for the same reason:
//! `Pool::restaged()` counts allocations, the deployment test requires
//! **nine** over six repeated decodes -- eight when the shape last changed
//! and one when the conversation crossed a page boundary -- and disabling the
//! reuse makes it sixty-six.
//!
//! What is left is `fire`, which is the GPU and 452 dispatches of it.
//!
//! ## And most of `fire` was the driver telling the card to stop
//!
//! Phase-profiled at last, a 6.45 ms decode of the real 4-bit qwen3-0.6b
//! divides like this: planning the 452 rectangles 0.85 ms, the one scalar
//! allocation 0.11, building and finding pipelines 0.25, writing 452
//! descriptor sets 0.12, recording 452 dispatches 0.37, and **inside the
//! submit-and-wait, 4.6**.
//!
//! That last number is not the model. A 0.6B decode is 452 dispatches over
//! six hundred megabytes of weights, and the card does that work in about
//! four fifths of a millisecond -- which is measurable, by recording the same
//! fire with the barriers removed: submit-and-wait falls from 4.6 ms to
//! **0.80**, and the whole step from 7.2 to 2.3. The other 3.8 ms was 451
//! `vkCmdPipelineBarrier`s at roughly eight microseconds each, which is what
//! it costs to drain and restart a compute pipeline between two dispatches
//! that take two microseconds apiece.
//!
//! A fire recorded that way is wrong, of course. The question is how many of
//! those 451 pairs are actually ordered, and the answer is that most are not:
//! a layer's Q, K and V projections read the same normed rows and write three
//! different places, the per-head rotary writes are disjoint, one norm's
//! scratch is not the next one's. What makes that visible is a decision this
//! driver made long before it had a reason to: it binds RANGES and never
//! `WHOLE_SIZE` -- see `Bound`, and the note there on why an overrun should
//! fault rather than read the next tensor -- so two rectangles of one arena
//! are two spans a recorder can compare.
//!
//! The missing half was which operands are WRITTEN. SPIR-V does not say
//! usefully: `slangc` decorates a buffer `NonWritable` only where the shader said
//! `readonly`, and these shaders mostly do not. The kernel table does say --
//! `kernels::Ty::BufMut` is "the launcher may write through this" -- and that
//! field had never been read by anything. `dispatch::Dispatch` now carries a
//! `writes` mask beside its `buffers`, built where a slot's index is still the
//! index of the operand that produced it, and `device::hazards` puts a barrier
//! in only for a real read-after-write, write-after-write or write-after-read
//! against everything recorded since the last one.
//!
//! Decode **7.2 -> 5.6 ms**, and the barriers 451 -> 311.
//!
//! Two smaller things went with it, both of the same shape as the module
//! cache beside them: `serve::fire` looked the kernel table's row up once per
//! LAUNCH, and `kernels::sig_in` is a linear scan of every row followed, for
//! a specialised name like `affine_qmm_t_bf16_gs_128_b_4`, by a second scan
//! of every row's axis points -- 0.23 ms a fire, for nine distinct symbols
//! asked 452 times. It is cached in the same map as the reflection now, so
//! the existing `Fired::parsed` assertion pins both: one miss, both lookups.
//! And the plan kept an owned `String` of each launch's symbol beside a
//! `Dispatch` that already borrowed it, which is 452 allocations for nothing;
//! that one is below the noise on this box and is recorded as a
//! simplification rather than a speedup.
//!
//! The claim needed two pins, because the two ways of being wrong are
//! opposite. Too many barriers is invisible -- the answers are right, only
//! slower -- so `Device::barriers()` counts them and the whole-plan test
//! requires strictly fewer than one per pair and more than none; a `hazards`
//! that always says yes reports 451 of 452 and fails. Too few is a race, and
//! races are the defects that pass: that pin is the byte-for-byte comparison
//! against `one_at_a_time`, which submits every dispatch on its own fence and
//! is the strongest ordering Vulkan has, over six texts and both fire
//! classes.
//!
//! One control did not fire and is recorded rather than dropped: removing the
//! write-after-read case entirely leaves every comparison passing on three
//! runs of all six texts, and changes the count on exactly one -- olmo2-1b,
//! 227 barriers to 211. It stays because the specification requires it, not
//! because this crate caught anything with it.
//!
//! ### What the suite says if the hazard analysis is wrong
//!
//! Both pins above were argued rather than measured, so the whole question
//! was, until later, closed only on speed. The measurement that closes it on
//! correctness is the ablation: neuter the emission site in `device.rs` to
//! `if false && at > 0 && hazards(..)` -- every barrier gone, the fastest the
//! backend can possibly be -- and **21 of the 69 device tests fail**. Among
//! them are the ones that matter most:
//! `both_real_models_agree_with_an_independent_implementation`,
//! `a_conversation_is_answered_the_same_however_it_reaches_the_driver`,
//! `the_cooperative_matrix_gemm_answers_what_the_baseline_one_does`, and
//! `fires_of_different_sizes_in_a_row_reuse_the_scratch_and_still_agree`.
//! So the 311 barriers are not a performance preference with an untested
//! safety story behind them: if `hazards` were to start saying no where it
//! should say yes, this suite says so, on this card, at this driver version.
//! That is weaker than a proof -- races are timing-dependent and a quieter
//! card might pass -- but it is the difference between a claim nothing checks
//! and one that 21 tests check.
//!
//! The tool that would check it properly is Vulkan's synchronization
//! validation, and it could not be turned on here. Neither
//! `VK_LAYER_KHRONOS_VALIDATION_VALIDATE_SYNC=true` nor a
//! `vk_layer_settings.txt` with `VK_LAYER_SETTINGS_PATH` produced a single
//! hazard report -- **including on the fully ablated build above**, which is
//! nothing but hazards. A tool that stays silent on a run with no barriers at
//! all is not reporting; it is off. Core and GPU-assisted validation do work
//! (see `fail_on_validation_error`), and syncval is recorded here as tried
//! and unavailable so the next person does not spend the afternoon on it
//! again.
//!
//! ## What the tier costs in accuracy, and the guard that hid the question
//!
//! Switching 146 modules on is a change of arithmetic, so it needs a claim
//! rather than a speedup. `tests/device.rs` makes one:
//! `the_cooperative_matrix_gemm_answers_what_the_baseline_one_does` runs the
//! real 4-bit qwen3-0.6b twice at the same tile, once with a module store
//! holding every `<symbol>.<tag>` key and once with them stripped, and
//! compares the logits. Over 151,936 of them the greatest difference is
//! **0.25 and the mean 0.036, against a largest logit of 25.6, with the same
//! argmax**. That is what `float16_t` A and B operands cost; the
//! dequantisation is bit-identical, as `qmm_t.slang` says, and the MULTIPLY is
//! not. It was written as an exact equality first, on the strength of that
//! header, and the header is right about the half it describes.
//!
//! Getting the two runs to differ at all took one more discovery. A 48-token
//! prompt at a tile of 32 produced answers equal to the last bit -- because
//! `llama_like/forward/mod.rs` puts the GEMM behind a `TokensMultipleOf(tile)`
//! guard, and 48 is not a multiple of 32, so **every projection took the GEMV
//! arm and the run never launched `affine_qmm_t` at all**. The symptom is the
//! same one this whole section started with: a comparison that refuses to
//! move. The prompt is 64 tokens now, and the test asserts the two runs
//! DIFFER before it checks by how much, so a future change that quietly
//! stops loading the tiered module fails here instead of passing quietly.
//!
//! And what the tier is worth end to end, rather than in accuracy: the same
//! 1024-token prefill at `(32, 32)` takes **570 ms** with the tiered modules
//! in the store and **2592 ms** with them stripped, which is the whole of the
//! 4.5x the tile change bought. The tile is not faster arithmetic. It is the
//! instantiation point at which the card's matrix units become nameable.
//!
//! `at_the_default_tile_the_tier_has_nothing_to_reach` is the control: the
//! same comparison at `(16, 32)` must come out exactly equal, because no
//! cooperative-matrix module is compiled at that tile. Both claims were
//! mutation-tested against a `Modules::code` that ignores the tier -- the
//! first fails, the second stays green.
//!
//! ## How to measure a shader change through the binary, since this got it
//! ## wrong once
//!
//! The first prefill A/B done here compared 41 s against 41 s and concluded
//! the attention rewrite did nothing. It was measuring the same SPIR-V twice.
//! `kernels-vulkan`'s build script only compiles the shaders under
//! `--features native`, and the feature changes the build-directory hash, so
//! `target/release/build/kernels-vulkan-<hash>/out/spv` names TWO different
//! directories and `cargo build -p pie` refreshes neither of them. A config
//! pointing at the wrong one is a stale-shader measurement that looks
//! exactly like a null result.
//!
//! The other half of the lesson: wall time through the whole binary varied
//! by seven seconds run to run on a shared box, which is wider than the
//! effect being looked for. The numbers above come from a harness that times
//! `Shell::step` alone.
//!
//! # The same claim, made through the front door
//!
//! Everything above is a device test calling `Shell::step`, which is this
//! crate measuring itself. So it was measured again through the real `pie`
//! binary -- a release build, a WASM inferlet, the engine, the gateway, the
//! whole path a user has -- with the memory preference switched off and on:
//!
//! | max tokens | host memory | device-local |
//! |---|---|---|
//! | 8 | 2.60 s | 4.55 s |
//! | 128 | 8.86 s | 6.33 s |
//! | 256 | 18.40 s | 11.11 s |
//!
//! 74.5 ms a token against 37.3, taken from the slope between the last two
//! so that start-up falls out. Two times exactly, end to end, which is the
//! claim the device tests make and this is the one that counts.
//!
//! The first row is the honest cost. Boot is about two seconds SLOWER,
//! because three hundred megabytes of weights now cross PCIe into
//! write-combined memory once instead of being left where the host wrote
//! them. Paid once per model, returned within the first fifty tokens.
//!
//! One more thing this measured, which was not what it set out to measure:
//! in a DEBUG build the same run takes 33 seconds either way -- 230 ms a
//! token with the memory preference on and 230 ms with it off. The driver is
//! not what is slow there, and neither is the GPU. Something on the host path
//! costs about 200 ms a token in an unoptimised build, and any future
//! end-to-end performance claim about this driver has to be taken from a
//! release binary or it is a measurement of `cargo build` instead.
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
//! That second finding had a sequel worth stating separately, because it was
//! a wrong answer rather than a crash. `try_reserve_exact` refusing was
//! reported as a FAULT, and so was a device that would not give the memory --
//! and neither is one. A growth that fails for want of memory is the
//! definition of [`frames::Launched::Exhausted`]: evict and re-post. That
//! variant was declared here, documented here, and matched on at the engine
//! seam, and **nothing in this crate had ever produced it**. Every full pool
//! took the fault path instead, which since a driver lane began answering its
//! token rather than hanging means the user's request dies for a condition
//! that clears the moment something else finishes.
//!
//! It is reachable in ordinary service, not only under abuse:
//! [`device::Device::budget`] reports a heap's SIZE, so the ceiling admits
//! any frame the device could hold **if it were empty**, and the device is
//! never empty -- the weights are in it. So [`device::Failed`] now separates
//! `OutOfMemory` from every other failure, `admit` turns the first into
//! `Exhausted`, and a refused growth is proven to leave the pool, the book
//! and the conversations in it exactly as they were -- which is what makes
//! the re-post correct rather than merely permitted.
//!
//! One arm of that is deliberately not proven on the device: the case where
//! the DEVICE, rather than the host, refuses. Provoking it means running a
//! shared machine out of memory to exercise a single comparison, so the
//! comparison is unit-tested on the Vulkan result codes instead, in both
//! directions -- an out-of-memory code must be retryable, and `DEVICE_LOST`
//! must not be, or a scheduler evicts and re-posts forever against a device
//! that is gone.
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
//! A channel whose producer has not run yet is `Filled::Early` -- not a fire,
//! and this crate used to call it not a fault either: its members were
//! published as RETRY, on the understanding that the scheduler would post
//! them again.
//!
//! It would not. Under ABI v14 the scheduler rejects a surviving RETRY
//! terminal by name -- "retry is not a v14 outcome (frame admission bounds
//! every in-frame gate)" -- and `worker`'s executor carries only SUCCESS and
//! FAILED. So the kind-looking word bought a worse error message and no
//! second attempt, and the seam now publishes FAILED with the reason logged
//! beside it. If admission really does bound every in-frame gate then this
//! arm is a broken invariant rather than a slow producer, and a broken
//! invariant should stop.
//!
//! Instrumenting a real `pie serve` never reached it, because the program
//! that fills a channel runs in the same call that reads it. The arm is kept
//! rather than deleted because `NotReady` is a state the resolver can still
//! return, and answering that with a fire would sample a distribution nobody
//! computed.
//! # The second class, and the three things it needed that were not there
//!
//! `DecodeEnvelope` is one of TWO device-resolved classes. The other is
//! `DeviceGeometry`, where the program traces its whole geometry -- pages, the
//! page CSR, the write descriptor, and a dense attention mask -- and the
//! driver READS what it traced instead of deriving it. This driver claimed
//! only the first, and a Track-B pass whose `AttnMask` binds a channel needs
//! `PIE_DEVICE_PORT_ATTN_MASK`, so the engine refused those passes at
//! classification and sent them down a host fallback that cannot derive
//! `EmbedTokens`.
//!
//! That one refusal was EIGHT of this tree's fifteen failing inferlets, and it
//! never named the mask: it said
//!
//! ```text
//! EmbedTokens is not host-derivable: channel N has no host-known value
//! ```
//!
//! which is the fallback failing, three layers below the claim that sent it
//! there. `RUST_LOG=info` named the real one.
//!
//! Widening the claim alone would have been the silent version of the bug --
//! the engine would send masks this driver stages as zeros. So the machinery
//! came first, and it was three things:
//!
//! * **the pitch.** `kernels-vulkan`'s `sdpa_paged_decode` wired
//!   `attention_mask_stride` to `Source::Slot(Kind::Param, 3)`, the model text's literal
//!   `0`; `kernels-wgpu` wires the same operand to the driver's own staged
//!   pitch. One word, and without it every mask is read at a stride of zero.
//! * **the rectangle.** [`resources::Frame::mask_from`] packs the fire's rows
//!   into one `[rows, widest]` rectangle with an enable byte each, and
//!   `Pool::stage` ships it. The pitch is the FIRE's widest row because one
//!   rectangle is bound for the whole fire; a per-request pitch reads every
//!   later row against another row's keys.
//! * **the branch.** [`envelope::fill`] now admits `DeviceGeometry` beside
//!   `DecodeEnvelope`: pages, last-page lengths and the mask are READ from the
//!   resolution rather than derived, and the dense mask is re-encoded as the
//!   same runs a host-lowered mask arrives in.
//!
//! Empty is not zeros, at every seam: an empty mask leaves the row's enable
//! byte clear and `attn/sdpa_paged.slang` applies the causal rule alone, while
//! a row of zeros forbids every key and produces a softmax over nothing.
//!
//! # Three walls, each visible only after the one before it fell
//!
//! Widening the claim moved the eight onto a second refusal, this one from
//! `pipeline::fire`: a frame slot may not consume a channel an earlier slot of
//! the same frame publishes when the earlier slot "resolves its descriptors on
//! the HOST". The rule was unconditional, and its own comment says it
//! describes CUDA's `FramePrepare`, which does every step's host work at frame
//! entry. It is false about this backend -- `launch` converts ONE step, fires
//! it, lets its program run, and only then converts the next -- so the answer
//! was a capability rather than an exemption: `resolves_geometry_per_step`,
//! default `false`, set `true` here alone.
//!
//! Behind that stood a third wall, and it was this driver's own:
//! `Frame::of` refused beam search's two lanes for both writing page 7.
//! Measured, they do: one instance, two lanes, both at position 0, both
//! tracing their first page. The refusal asks a SCHEDULER's question -- did
//! two requests get placed on one page -- of requests the scheduler did not
//! place, and the derivation it asks it with cannot tell two lanes apart at
//! all, because it reads the write offset off the POSITION.
//!
//! So the write descriptor stopped being a claim this driver made and did not
//! read. `envelope::fill` reads `Port::WSlot`/`WOff`, translates the pages,
//! and carries them beside the plan; [`resources::Request::writes`] holds them
//! per row; `Frame::of` uses them where they are stated and its own division
//! where they are not. The intermediate version -- keep the derivation, CHECK
//! the statement against it -- is worth recording because it is what proved
//! the derivation wrong, in one sentence, on the second fire of every beam:
//!
//! ```text
//! instance 1 request 1 writes position 1 to page 7 offset 2, and its own
//! span places that position at page Some(7) offset 1
//! ```
//!
//! Two lanes share the page they forked from and take separate SLOTS inside
//! it. The statement is the answer; the derivation was the wrong question.
//!
//! The sweep went from 24/39 inferlets to 33/39. What is left is four
//! model-gated intrinsics (`attn_score`, `envelope_dot`), which `driver-metal`
//! does not serve either, and two failures with separate causes.
//!
//! # A field added to `Request` is a field `slice` can drop
//!
//! [`turns::slice`] cuts a fire into sub-fires and rebuilds each piece with
//! `Request::of`, which knows positions and pages. Every field added since is
//! one that walk drops in silence, and two already were: `traced`, which put
//! beam search's lanes straight back on the page-sharing refusal, and the mask
//! rows, which came out WHOLE -- read against the piece's own row numbering,
//! which is another row's allow-bytes. Both are cut with the rows now, and
//! `a_piece_of_a_request_carries_its_rows_mask_and_write_targets` is there so
//! the next field added is refused by a test rather than dropped by a helper.
//!
//! Module map, invariants and measurements: `.wiki/driver-vulkan.md`.

// Not the workspace lint table: that one forbids `unsafe_code`, and every `ash`
// entry point is unsafe.
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout)]

#[cfg(feature = "native")]
pub mod hold;
#[cfg(feature = "native")]
pub mod bind;
#[cfg(feature = "native")]
pub mod binding;
#[cfg(feature = "native")]
pub mod device;
#[cfg(feature = "native")]
pub mod dispatch;

#[cfg(feature = "native")]
pub mod encode;
#[cfg(feature = "native")]
pub mod envelope;
pub mod facts;
#[cfg(feature = "native")]
pub mod frames;
pub mod geometry;
// `pub mod lowering` STOOD HERE -- 539 lines that read a `KernelSig`'s
// operand kinds and answered which of a call's values was a descriptor and
// which a push field, plus the `Call`, `Value` and `Mismatch` vocabulary for
// saying so. Its production caller was `dispatch::plan_one`, which is gone,
// and its last caller of any kind was one GPU test that now transcribes the
// twenty-four bytes `attn/kv_write.slang` declares. What packs a routine's
// scalars is `binding::params_from` feeding `encode::Encoder`.

// Strings and a static table, so it is ungated: a caller that only wants to
// know what a checkpoint calls `layer.3.down` should not have to link Vulkan.
//
// It LIVED here, and `driver-wgpu/src/names.rs` was a byte-for-byte copy of
// the same 412 lines -- the second hand-written copy of one golden table,
// which is the failure `driver`'s own module documentation opens by naming.
// It is one table because it describes ONE thing, the names
// `model::boot::compile_load_plan_for` publishes; two shells reading one
// producer is not two conventions that happen to agree.
pub use driver::names;
// Pure Rust, and gated only because it speaks in `resources`' `Shape` and
// `Request` -- which are pure data in a module that also holds device
// handles. Splitting that module to ungate this one would buy nothing today.
#[cfg(feature = "native")]
pub mod pages;
#[cfg(feature = "native")]
pub mod replay;
#[cfg(feature = "native")]
pub mod resources;
// Wall-clock accounting for the HOST side of a step. Ungated: it holds no
// device handle and a caller with no GPU can still read the totals.
pub mod phase;
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
pub use spirv::Declared;
