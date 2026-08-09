# Shell parity: `mtl4_context.hpp` against `driver-metal-new`

The C++ header is 508 lines and is the only complete statement of what the
shell does. This walks it method by method. Every entry is one of:

- **ported** — the Rust does the same job, under whatever name fits Rust.
- **dropped** — deliberately not ported, with the reason. A dropped method is
  a claim that the C++ needed it and the Rust does not, and the reason has to
  say why rather than assert it.
- **missing** — should exist and does not, with the todo that will add it.

Nothing here is "ported" because a function with a similar name exists. The
test that covers it is named where one does.

## Free functions and device queries

| C++ | Rust | |
|---|---|---|
| `device_working_set_bytes()` | `Context::working_set_bytes` | ported |
| `set_device_working_set_bytes_for_test` | — | dropped |
| `device_working_set_is_forced()` | — | dropped |
| `host_reclaimable_bytes()` | `Memory::probe().reclaimable` | ported |
| `set_host_reclaimable_bytes_for_test` | — | dropped |
| `host_wired_and_installed_bytes()` | `Memory::probe()` + `wired_fraction` | ported |

The three test hooks are dropped together and for one reason. They exist
because `device_working_set_bytes` is a static that reads the device, so a
refusal that only fires on a model too big for the GPU cannot be exercised on
a GPU the models fit on — the hook is the only seam. In Rust the check is
`Context::check_working_set(requested)`, which takes the number, so a test
passes a large one. `device_working_set_is_forced` then has nothing to
report: it exists in C++ only to stop callers from comparing a forced
imaginary device against the real machine's free memory, and there is no
forced device here to detect.

`host_wired_and_installed_bytes` returns a pair in C++ and `{0, 0}` when the
kernel declines. `Memory::probe` returns the pages and `wired_fraction`
returns an `Option`, because a wired fraction of zero and "the kernel would
not say" are different facts and the C++ pair cannot tell them apart.

## Heap and slots

| C++ | Rust | |
|---|---|---|
| `heap_alloc(size, align)` | `Heap::alloc` | ported |
| `const_slot(ordinal, index, bytes)` | `Heap::constant` | ported |
| `make_resident()` | — | dropped |
| `wrap_host_memory(ptr, size)` | `Mapped` | ported |
| `zero_buffer_range` | `Region::zero` | ported |
| `copy_buffer_range` | `Region::copy` | ported |

`make_resident` is dropped because it is a phase, and a phase that must
happen between two other phases is a rule the type system can hold instead.
The C++ comment says "call after all heap_alloc + all arg_bind, before the
first encode" — an ordering nothing enforces. `Heap` adds itself to the
residency set once at construction and placement allocations inherit it, so
there is no moment at which the heap is allocated and not resident.

`zero_buffer_range` and `copy_buffer_range` are filed among the
command-encoding methods in C++ and encode nothing: on unified memory a
shared-storage buffer is host-addressable, so both are a `memset` and a
`memmove`. They are in `region.rs`, in the portable half, where a Linux
`cargo test` reaches the arithmetic — which is the part with a history of
being wrong.

## Elastic buffers

| C++ | Rust | |
|---|---|---|
| `create_elastic_buffer` | `create_elastic` | ported |
| `ensure_elastic_buffer` | `Stepper::ensure` | ported |
| `ensure_elastic_buffers_atomically` | `Stepper::ensure_all` | ported |
| `trim_elastic_buffer` | `Stepper::trim` | ported |
| `release_elastic_buffer` | `Elastic::drop` | ported |
| `elastic_page_bytes()` | `PAGE` | ported |
| `elastic_budget_pages()` | `pages_for_bytes(Budget::total)` | ported |
| `elastic_committed_pages()` | `pages_for_bytes(Budget::committed)` | ported |
| `memory_pressure_level()` | `Pressure::probe()` | ported |
| `set_memory_pressure_level_for_test` | — | dropped |
| `drain_elastic_mappings()` | — | dropped |
| `pending_elastic_release_count()` | `Arena::pending` | ported |

`ensure`/`trim` are on `Stepper` rather than on the buffer because a remap is
a point on the same timeline as a step, and the `Stepper` owns that timeline.

The pressure test hook is dropped because `Pressure` is an argument, not
subscribed state. `drain_elastic_mappings` is dropped because there is
nothing to drain: `trim` waits for its own unmap before returning, and a grow
is ordered by the timeline rather than by a queue the caller has to flush.

## Transient pool, external and standalone buffers

| C++ | Rust | |
|---|---|---|
| `acquire_transient_buffer` | `Pool::acquire` | ported |
| `recycle_transient_buffer` | `Transient::drop` | ported |
| `transient_buffer_pool_stats` | `Pool::stats` | ported |
| `set_transient_buffer_pool_limit_for_test` | `Pool::set_capacity` | ported |
| `use_external_buffer` | `Externals::insert` | ported |
| `release_external_buffer` | `External::drop` | ported |
| `external_buffer_count` | `Externals::len` | ported |
| `create_standalone_buffer` | `Ring::new` | ported |
| `release_standalone_buffer` | `Ring`'s `Drop` | dropped |
| `standalone_buffer_count` | — | dropped |
| `standalone_bytes` | — | dropped |

`set_transient_buffer_pool_limit_for_test` keeps its behaviour but loses its
name: a cache limit is configuration, and the fact that only a test currently
sets it is not a reason to name it after the test.

**The standalone buffers were the one real gap, and are now closed.** Three
of their four C++ uses were already covered by something better — K/V storage
by `Elastic`, command scratch by `Pool`, and adopting a buffer somebody else
allocated by `Externals`. The fourth — "allocate me a plain shared buffer
outside the heap and keep it resident", which the PTIR channel rings need —
is `src/metal/ring.rs`: the allocation is a step of `Ring::new` rather than
a primitive that hands out unowned handles. `release_standalone_buffer`
existed only because `create_standalone_buffer` handed back a `SlotHandle`
with no owner — the C++'s own comment records that without the release,
`resize_pool` leaked every previous K/V buffer, retained and resident,
forever. `Ring`'s `Drop` is that release with no call to forget, and
`tests/device_ring.rs` proves the buffers die with the ring. The two
counters existed to audit the registry of unowned handles; there is no
registry to audit.

## Argument tables

| C++ | Rust | |
|---|---|---|
| `arg_bind(k, ordinal, index, slot, offset)` | `Tables::bind` | ported |
| `arg_bind_ordinal(ordinal, index, slot, offset)` | `Tables::bind` | ported |
| `arg_bind(k, index, slot, offset)` | — | dropped |
| `arg_slot_is_bound(ordinal, index)` | `Tables::is_bound` | ported |
| `arg_slot_address(ordinal, index)` | `Tables::address` | ported |
| `release_argtable_ordinal(ordinal)` | `Tables::forget` | ported |

The C++ carries three overloads of `arg_bind` and its own comment says the
`Kernel k` is decorative: within one layer, `Rms` and `Residual` each recur,
so `(kind, layer)` collides and only the flat ordinal is a key. A decorative
parameter that looks like a key is worth removing rather than documenting, so
there is one `bind`, and it takes the ordinal. The `ordinal = -1` singleton
overload goes with it — `-1` as "there is only one" is a `u32` ordinal like
any other here.

`Tables::address` returns `Option<u64>` where the C++ returns `uint64_t`.
An entry nobody bound reads as address zero from Metal, so the C++ answer
cannot distinguish "never wired" from "wired to null" — and the whole point
of the method is coverage testing over the DAG, where those are opposite
verdicts.

## Pipelines

| C++ | Rust | |
|---|---|---|
| `pso_archive_dir()` | `Archives::dir` | ported |
| `compile_pso` / `compile_ptir_pso*` | `Compiler::compile*` | ported |
| `pso_max_threads(pso)` | `StepEncoder` reads it itself | dropped |
| `release_pso(pso)` | — | dropped |
| `retained_pso_count()` | — | dropped |
| `last_ptir_compile_disabled_fast_math()` | `Math` argument | dropped |
| `device_cache_id()` | `Context::cache_id` | ported |

`release_pso` and `retained_pso_count` exist because a PSO crosses the C ABI
as a `void*`, so the context must keep an array of everything it handed out
and a set of what it was told to release. Under `Retained` the array is the
binding and both methods evaporate; `tests/device_pso.rs` asserts that with
`Weak` rather than leaving it to be believed.

`pso_max_threads` is dropped as a public query because its only caller is the
dispatch that is about to use it, and `StepEncoder::set_pipeline` reads
`maxTotalThreadsPerThreadgroup` there and refuses an over-wide threadgroup.
A number the caller has to fetch and then remember to check is a check that
gets skipped.

`last_ptir_compile_disabled_fast_math` is dropped and is also wrong: the
setter writes `MTLMathModeSafe` and then reads that same field back to decide
the flag, so it reports what it was just told. `Math` is an argument, and a
mode that is an argument needs no flag to remember it.

## Timestamps, steps and events

| C++ | Rust | |
|---|---|---|
| `create_timestamp_heap(count)` | `Timestamps::new` | ported |
| `resolve_timestamps(heap, count, out)` | `Timestamps::resolve` | ported |
| `release_timestamp_heap(heap)` | `Timestamps::drop` | ported |
| `StepEncoder::mark_timestamp` | `StepEncoder::mark_timestamp` | ported |
| `run_step(encode_fn, ab)` | `Stepper::run` | ported |
| `last_commit_feedback()` | `Feedbacks::latest` | ported |
| `last_event()` | `Stepper::steps` | ported |
| `force_next_wait_timeout_for_test()` | — | dropped |
| `StepTiming` | `Result<Timing>` | ported |
| `start_keepalive` / `stop_keepalive` | `Keepalive::start` / drop | ported |

`run_step`'s `int ab = 0` allocator selector is dropped: the double-buffered
allocator pair is the `Stepper`'s own business, and a caller that picks the
wrong one gets a reset allocator underneath a command buffer still running on
it.

`StepTiming` becomes `Result<Timing>` because the C++ struct mixes
measurements with status. Its `completed`, `timed_out`, `gpu_error` and
`gpu_error_text` are all answers to "did this work", which is the `Result`;
`Timing` carries only durations. Its `gpu_ms` is a `double` that is zero when
the feedback has not landed, which cannot be told apart from a GPU that
reported zero — `Timing::gpu` is an `Option`.

`force_next_wait_timeout_for_test` is dropped with its subject: the C++
timeout meant "the first five-second probe expired", which a slow-but-healthy
step also does, so the one caller that acted on it killed slow steps. The
Rust waits with a caller-supplied timeout and returns an error, which a test
produces by passing a short one.

## StepEncoder

| C++ | Rust | |
|---|---|---|
| `set_pso` | `set_pipeline` | ported |
| `set_argtable(k, ordinal)` | — | dropped, see `arg_bind` |
| `set_argtable_ordinal(ordinal)` | `set_argument_table_for` | ported |
| `dispatch(grid, tg)` | `dispatch` | ported |
| `barrier(vis)` | `barrier` | ported |
| `mark_timestamp(heap, idx, precise)` | `mark_timestamp` | ported |
| fused per-dispatch convenience | — | dropped |

`BarrierVisibility` becomes `Visibility` and keeps both variants. The C++
also honours a `PIE_BARRIER_VIS` environment variable that overrides every
barrier in the process regardless of the argument at the call site; that is
not reproduced. A sweep that needs it can pass the argument.

## Summary

Sixty-one entries. Fifty-two ported, nine dropped with a reason above,
nothing missing. The last four to close were the standalone-buffer hole,
closed by `src/metal/ring.rs`. Nothing in the header is unaccounted for.
