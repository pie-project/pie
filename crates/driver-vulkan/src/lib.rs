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
//!   driver reads one row of. See the note beside `widest_readout`: the guest
//!   sees `driver published poison epoch 1`, which names nothing, and the
//!   reason is only in the engine's WARN log.
//!
//! So of the failures, none is a Vulkan kernel, none is a binding, and none
//! is this driver's lowering. That is the claim the sweep was worth making,
//! and it is a different claim from any test passing.
//!
//! # The one capability `driver-metal` has that this does not, and its price
//!
//! The sweep answered what this backend REFUSES. The other half of the
//! question is what it does not advertise, and comparing the two seams'
//! `DriverCapabilities` field by field leaves exactly one real difference in
//! `driver-metal`'s favour: it is ELASTIC -- `elastic_page_bytes` is a page
//! and `elastic_budget_pages` is a real number -- and this is not. (The
//! difference in the other direction is `kv_copy_domain_mask`, where this
//! backend advertises device-to-device and Metal advertises nothing.)
//!
//! The interesting part is that `Shell::resize_pool` here is not missing. It
//! is written, it preserves the pages that survive, it refuses by name a
//! shrink that would strand a seated conversation, it leaves the pool intact
//! when the machine will not stage the new one, and all of that is proven on
//! the device by
//! `a_cache_resized_under_a_conversation_does_not_change_its_answer`. It is
//! working code that production never reaches, because `bootstrap` starts a
//! trim task only when both elastic numbers are non-zero and both are zero.
//!
//! The seam gave a reason for the zero, and the reason was false: "nothing
//! can be given back page-wise". A shrink here frees the old buffers and
//! takes smaller ones, so bytes do come back, at whatever granularity the
//! caller names. Right answer, wrong reason -- the same shape of defect as
//! the mask above, and the only way to tell the two apart is to measure.
//!
//! So it was measured. `Pool::resize` reads every layer's whole old buffer
//! down to host memory and writes a fresh one back up, so the charge is the
//! pool's size twice and the delta does not enter it. At 256 pages of
//! qwen3-0.6b, handing back ONE page takes 2.77 s; handing back a hundred
//! and twenty-six takes 0.74 s. The deeper cut is nearly four times cheaper,
//! because the destination it fills is smaller. **The cheapest trim this
//! pool offers is the largest one**, which is the opposite of what a trim
//! task is for -- and a resize peaks at both sizes at once, so a shrink
//! asked for under memory pressure needs more memory than not shrinking.
//!
//! Zero is therefore the honest advertisement, and it now sits next to the
//! measurement instead of next to a guess.
//! `giving_back_one_page_costs_what_giving_back_half_the_pool_costs` holds
//! the floor: it goes red the day this pool becomes genuinely page-wise --
//! Vulkan sparse binding behind a `sparseResidencyBuffer` tier is how that
//! would happen -- and the red is the reminder to come back and advertise
//! what would by then be true.
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
//! `attn/sdpa_paged.comp` runs one thread per head dimension, and every one
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
//! "After" is `sdpa_paged.comp`'s tiled path made cooperative the same way
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
//! The second half is not this crate's to take. The tile comes from
//! `QMM_TILE`, one `const` in `crates/model` shared by every backend, and
//! changing it means touching 38 `MetalBinding` constructors across 22 files
//! including two other drivers. What is worth handing to whoever owns it is
//! the number above and one observation: **`kernels-metal` stamps no `bm_16`
//! at all** -- its tiles are 32, 64 and 128, all carrying `_wm_/_wn_`
//! suffixes -- so `(16, 32)` is a point that resolves in the Vulkan and wgpu
//! trees and nowhere else. It reads much more like the Vulkan stamp than
//! like a Metal one, which makes "widen it" a smaller question than the
//! constant's shared home suggests.
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
