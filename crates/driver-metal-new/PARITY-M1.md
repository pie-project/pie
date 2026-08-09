# Pipeline parity: `pipeline/m1_runtime.cpp` against `driver-metal-new`

The C++ is 3411 lines plus a 202-line header, and it is the only complete
statement of what the launch path does. Same rules as `PARITY.md`: every entry
is **ported**, **dropped** (with a reason that says why the C++ needed it and
the Rust does not), or **missing** (with what is blocking it). Nothing is
"ported" because a function with a similar name exists.

The port is in progress. The portable half — everything that is a function of
the plan and the fire's numbers rather than of the device — goes first, because
it is the half that can be tested without a GPU and it is where the C++'s
mistakes are.

## Cache identity — `src/pipeline/identity.rs`

| C++ | Rust | |
|---|---|---|
| `encode_m1_cache_identity` | `cache_identity` | ported |
| `encode_cache_identity` | — | dropped |
| `M1CacheIdentityVersions` | `Versions` | ported |
| `combined_signature` | `combined_signature` | ported |
| `fnv1a64` | `tensor_ir::fnv1a64` | dropped |
| `hex64` | — | dropped |
| `identity_bytes` | — | dropped |

`encode_cache_identity` is the two-argument wrapper that filled `Versions` from
`PTIR_COMPILER_VERSION` and friends. Those live in the compiler's headers and
this crate deliberately does not depend on the compiler, so the versions are a
parameter and the fill belongs to whoever assembles the driver. The emitter
version in particular must come from `ProgramRegistration::emitter_version`
rather than a driver-side copy: the C++'s copy said 23 while the host said 36.

`fnv1a64` is dropped because `tensor-ir` already owns it and the CUDA driver and
host program cache reach the same number through it. `hex64` and
`identity_bytes` are `format!("{:016x}")` and `u64::to_le_bytes`.

## Value shapes — `src/pipeline/extent.rs`

| C++ | Rust | |
|---|---|---|
| `M1RuntimeExtents` | `Extents` | ported |
| `symbolic_extent` | `Role` + `Extents::get` | ported |
| `describe_value` | `describe` | ported |
| `DeviceValueDesc` | `ValueDesc` | ported |
| `value_bytes` | `ValueDesc::device_bytes` | ported |
| `wire_value_bytes` | `ValueDesc::wire_bytes` | ported |
| `m1_extents_from_forward_desc` | `batch::ForwardDesc::extents` | ported |
| `m3_extents_from_forward_desc` | `batch::ForwardDesc::extents_from_readout` | ported |
| `resolve_m1_shape_for_test` | — | dropped |
| `M1ResolvedShape` | — | dropped |

The two `*_from_forward_desc` constructors were field copies out of
`batch::MemberForwardDesc`, which had no Rust counterpart when this section
was written; they landed with the `batch/` port as methods on
`batch::ForwardDesc`, the type that owns the fields.

`resolve_m1_shape_for_test` and its `M1ResolvedShape` exist because
`describe_value` is in an anonymous namespace and a test cannot reach it. It is
`pub` here and tested directly, so the hook has nothing to do.

Three C++ behaviours are refusals here: an unrecognised extent role (was
`return 1`), a rank past four (was silent truncation, which drops a factor from
the element count and under-sizes every allocation derived from it), and a
32-bit `len * 4` that reported zero bytes for a value of 2^30 f32 lanes.

## Scratch layout — `src/pipeline/scratch.rs`

| C++ | Rust | |
|---|---|---|
| `align_up` | `align_up` (private) | ported |
| `kMaxScratchBytes` | `MAX_BYTES` | ported |
| the value-offset loop in `execute` | `layout` | ported |
| the `dummy` subhandle at offset 0 | `DUMMY_BYTES` | ported |

The C++ accumulated the total with unchecked `+=` and tested the bound after,
so a wrapped total passed a check the real one would have failed. Every step is
checked here. The placeholder descriptor the C++ pushed onto an empty list is
not part of the layout — it is what keeps the *buffer allocation* non-empty, so
it belongs at the allocation site.

## Op parameters — `src/pipeline/params.rs`

| C++ | Rust | |
|---|---|---|
| `DeviceOpParams` | `OpParams` | ported |
| the record fill in `execute` | `OpParams::of` | ported |
| the record fill in the M2 path | `OpParams::of` | ported |
| `op.args.size() > 1 \|\| tag == PIVOT_THRESHOLD` | `binds_second_argument` | ported |

The C++ wrote the record twice — once in the M1 `execute` loop and once in the
M2 command builder — with the two copies agreeing by inspection. One function.
`sink_bytes` stays zero: it comes from the bound channel cell, not from the op.

## Readiness — `src/pipeline/readiness.rs`

| C++ | Rust | |
|---|---|---|
| `check_readiness_host` | `check` | ported |
| — | `Words`, `check_words` | added |
| `M1PrepareOutcome` | `Readiness` | ported |
| `M1ChannelEffect` | `Effect` | ported |
| `batch::ChannelTicket` | `Ticket` | ported |
| `kNoTicket` | `NO_TICKET` | ported |

Every outcome in the C++ was a string with the channel index and the failure
kind encoded as arithmetic (`0x200 + channel` permanent, `0x300` early, `0x500`
an unorderable put). Nothing parsed them back, so the distinction was lost the
moment it was made. `Reason` names each case and `is_permanent` answers the
question the base addresses were encoding.

`Words` and `check_words` were added when the device ring arrived: the check
is a function of four ring words, and `ChannelState` was its first caller,
not its input. `check` is now the snapshot of a `ChannelState` fed into
`check_words`, and `metal::Ring::snapshot` feeds the same check instead of a
second copy of it.

## M3 grouping — `src/pipeline/group.rs`

| C++ | Rust | |
|---|---|---|
| `m3_schedule_bucket` | `schedule_bucket` | ported |
| `m3_stage_key` | `GroupKey::of` | ported |
| `M1Runtime::m3_stage_group_key` | `GroupKey::of` | ported |
| `m3_used_channel_slots` | `used_channel_slots` | ported |
| `m3_channel_flags` | `channel_flags` | ported |
| `kM3Channel*` | `CHANNEL_*` | ported |
| `kMetalM1MaxChannels` | `MAX_CHANNELS` | ported |

The key was a `reinterpret_cast` of a `u64` into a `std::string` with a byte
pushed on; it is two numbers. "No key" was the empty string, which is itself a
usable map key, from three different causes; it is `None`.

`used_channel_slots` gains the bound the C++ applied to the declared channel
count and not to this one, even though this is the count that gets bound.

## Compile cache — `src/pipeline/cache.rs`

| C++ | Rust | |
|---|---|---|
| `programs` / `stage_cache` / `negative` | `Bounded` | ported |
| `kMaxProgramCacheEntries` etc. | `MAX_*_ENTRIES` | ported |
| `M1CompileFailureKind` | `Failure` | ported |
| `remember_negative` | `Bounded::insert` | ported |
| `M1CacheStats` | `Stats` | ported |
| `set_program_cache_capacity_for_test` | `Bounded::new` | dropped |
| `inject_stage_cache_entry_for_test` | `Bounded::insert` | dropped |

The two test hooks are dropped because the thing they inject into is a public
type with a public constructor here; the C++ needed them because the caches
were private members of an `Impl` behind a pimpl.

The behaviour change is the point of the slice: the C++'s positive caches never
evicted, and a full one returned a *retryable* failure — so the sixty-fifth
distinct program a process saw could never run, and the caller retried forever
against the one condition retrying cannot change. The negative cache evicted
`begin()`, which is neither the oldest nor the coldest entry. All three are LRU.

## Bind-time derivation — `src/pipeline/meta.rs`

| C++ | Rust | |
|---|---|---|
| `collect_singleton_metadata` | `op_metadata` | ported |
| `M1OpMeta` | `OpMeta` | ported |
| the inline walk in M2 validate | `op_metadata` | dropped |
| the inline walk in the M2 builder | `op_metadata` | dropped |
| the inline walk in the M3 builder | `op_metadata` | dropped |
| the `effects.resize` loop | `channel_effects` | ported |

The three dropped entries are the same running sum written out by hand at
three more call sites, each maintaining its own `result_base` local. They are
dropped in the sense that they do not become three Rust functions.

`M1OpMeta` carried a by-value copy of the whole `PlanOp` — a struct with two
vectors — so binding duplicated the op list next to the list it walked. The
Rust holds `node` and reads the op through it.

The walk gains two refusals. The base accumulated in an unchecked `uint32_t`,
and a wrapped base is not a large index that a later bounds check catches but a
small one that passes and aliases another op's results. And the header states
that the walker "assumes the plan is well-formed", justified by the host
validating first — true of the path the host emitted, not of a plan arriving
over the ABI, and the check is one comparison.

`channel_effects` gains a consistency check between a channel's declared
readiness and the ops that touch it. `PIE_READINESS_UNTOUCHED` on a channel
something takes means the gate the take needs was never computed, and the C++
would run the take against a ring it never checked was non-empty. A capacity of
zero is full and empty at once, so both gates are unsatisfiable; the C++
defaulted the field to 1 and then overwrote it with the plan's zero.

## Device status — `src/pipeline/status.rs`

| C++ | Rust | |
|---|---|---|
| `DeviceStatus` | `Status` | ported |
| the M1 status decode (1945–1968) | `Outcome::of` + `report` | ported |
| the M2 status decode (2385–2408) | `Outcome::of` + `report` | dropped |
| the M3 lane report (3169–3220) | `Outcome::of` + `report` | ported |
| the `site ==` chain | `Site` | ported |
| `static_assert(sizeof(DeviceStatus) == 16)` | `STATUS_BYTES` + a test | ported |
| — | `FAULT_CLASSES`, `describe_fault` | added |

Three copies of the decode, agreeing on `state == 4` and `state == 2` and on
nothing else. The M1 copy printed the fault in decimal — `160` for a code the
whole rest of the system writes as `0xA0` — and discarded `reserved0` and
`reserved1`, so the guard site the kernel deliberately recorded was thrown
away. The M3 copy printed hex and decoded the site. Same kernel, same fault,
two reports.

All three treated "not 4 and not 2" as an op fault, which swallows `state = 0`
(the kernel wrote nothing) and `state = 1` (the kernel started and stopped)
into "generated op fault 0". The M3 path had learned half of this — it guards
on `encoded`, because a group that never dispatched reads back as a lane-wide
zero fill and produced a GPU fault report for something the GPU was never asked
to do. The M1 path never learned it. `Diagnosis` separates all four.

`describe_fault` is new. `codegen/fault.rs` declares every code a kernel can
write, with the per-channel classes and the two that alias op tags, and its own
module doc says "Nothing decodes these: the drivers surface the number and a
human reads it". The table exists; the driver may as well read it. The mirror
is checked against `tensor_compiler::codegen::fault::CLASSES` in a test, with
the compiler as a dev-dependency only, so the copy cannot drift.

## Stage cache and its collision guard — `src/pipeline/stage_cache.rs`

| C++ | Rust | |
|---|---|---|
| `Impl::stage_cache` | `Stages` | ported |
| `pending_stages` | `Stages` pending half | ported |
| the guard against `stage_cache` | `Stages::lookup` | ported |
| the guard against `pending_stages` | `Stages::lookup` | dropped |
| `M1StageExecutable::stage_identity` | `Entry::identity` | ported |
| `identity_bytes` | — | dropped |
| `default_m1_cache_dir` | `metal::Archives::discover` | dropped |

Keying a stage on a hash and storing a second, independent identity beside it
to check after a hit is the right design and the C++ had it. What it did with a
detected collision is the defect: `reject_deterministic`, which is the
classification that says *this program* cannot compile and never will, and is
the classification the negative cache remembers. A collision is not a property
of the program being compiled — it is a property of which other program holds
the slot. The C++ blamed a program for a collision it did not cause and then
wrote the verdict down. A collision here evicts the incumbent and returns a
miss, and `Stages::collisions` counts it so the rate stays visible.

The guard was written out twice, identically, once for each map; the pending
map exists so a compile that fails partway leaves the cache untouched, which is
`commit`/`abandon` here rather than a second map with a second copy of the
guard. `stage_identity` was a `std::vector<std::uint8_t>` holding a `u64`'s
eight bytes, heap-allocated per entry to compare a number.

The capacity check was the program cache's mistake again — `size() + pending >=
max` returning a retryable failure — and is gone for the same reason.

`default_m1_cache_dir` is dropped because `metal::Archives::discover` already
does it, and does it better: the C++'s last resort was
`return ".pie-metal-ptir-cache"`, a *relative* path, so a process started
without `HOME` scattered a compile cache into whatever directory it happened to
be launched from. `Archives` has no cache at all in that case, which is the
honest answer.

## Emitted-kernel index — `src/pipeline/emitted.rs`

| C++ | Rust | |
|---|---|---|
| `HostEmittedKernels` | `Emitted` | ported |
| `HostEmittedKernels::find` | `Emitted::get` | ported |
| `HostEmittedKernels::Key` | the map's tuple key | ported |
| `HostEmittedKernels::KeyHash` | — | dropped |
| the `error`-before-`source` convention | `Slot` | ported |

`emplace` on an `unordered_map` keeps the entry already present and drops the
new one, silently. So a host that emitted two kernels for one `(kind, stage,
region)` got whichever came first in the vector — a choice between two kernels
made by array order, by a driver with no way to know which the host meant, and
if the two differ at all one of them is wrong. `Emitted::index` reports it.

The three states `EmittedKernel` packs into two strings become `Slot`'s
variants. The C++ has them right and says so in a comment on the container:
callers must read `error` before `source`, because an empty source with a
populated error is a *deliberate* refusal that the driver answers with its
slower path rather than a failure. That comment is not next to any of the call
sites that must obey it. The order is inside `get` here. `Slot::Malformed` is
the fourth state the C++ had no name for — both strings empty, which `find`
returned like any other entry and the caller compiled as `""`.

`KeyHash` is dropped rather than ported: it packed `stage << 24` over a
full-width `region`, so `(stage 1, region 0)` and `(stage 0, region
0x0100_0000)` hashed alike. The map compared full keys, so this cost lookups
rather than correctness — but it is a hand-written hash with a bug in it and
the standard one has neither.

## Already covered elsewhere in the crate

| C++ | Rust | |
|---|---|---|
| `inline_ptir_rng_preamble` | `shader::splice_with` | dropped |
| `kPtirRngInclude` | `shader::DIRECTIVE` | dropped |
| `default_m1_cache_dir` | `metal::Archives::discover` | dropped |
| `fnv1a64` | `tensor_ir::fnv1a64` | dropped |
| `align_up` | `scratch` (private) | dropped |
| `wire_value_bytes` | `pipeline::wire_cell_bytes` | dropped |

`inline_ptir_rng_preamble` is a `find`/`replace` loop over the literal text
`#include "ptir_rng.generated.metal"`, anywhere it appears, mutating the string
under the cursor it is scanning with. `shader::splice_with` was already written
against the same requirement and is stricter in the two ways that matter: it
honours a directive only at column zero, so the same characters inside a
comment or a string literal are left alone, and it builds the output forward
so the scan never revisits text a replacement introduced. It also handles
nested includes and bounds the depth, neither of which the C++ attempts.

## The struct zoo — `src/pipeline/lane.rs`, and where the rest went

The C++ declares fifteen structs in one block (lines 388–546) because a single
translation unit wants every type before any function. They are not one kind
of thing, and in Rust they do not become one module. What the three paths
actually *share* is the lane table and its grouped sidecars — data with an ABI
— and that is what this slice ports, portably. The executables, the fire and
the command plans are each the output of exactly one function, and each lands
with its builder, where its invariants are established.

| C++ | Rust | |
|---|---|---|
| `PtirLaneTableHeader` (use) | `lane::Header` | ported |
| `PtirLaneRecord` (use) | `lane::Record` | ported |
| `PtirLaneChannelSlot` (use) | `lane::ChannelSlot` | ported |
| `M3ChannelMeta` | `lane::ChannelMeta` | ported |
| `M3GroupLayout` | `lane::GroupLayout` | ported |
| `M3RowMeta` | `lane::RowMeta` | ported |
| the `lane_bytes` formula, twice | `lane::Shape::bytes` | ported |
| the header/record/slot pointer walks, twice | `lane::Shape` offsets | ported |
| `static_assert(sizeof(...))` ×3 (651–653) | size **and offset** tests | ported |
| `M1ProgramExecutable::grouped_reason` | — | dropped |
| `M1StageExecutable::cache_identity` | — | dropped |
| `M1StageExecutable::stage_identity` | `Stages`'s entry | dropped |
| `M1RegionExecutable`, `M2FusedRegionExecutable`, `M3GroupedRegionExecutable` | `metal::program::{RegionExecutable, FusedExecutable, GroupedExecutable}` | ported: `compile_program` |
| `M1StageExecutable`, `M1ProgramStage`, `M1ProgramExecutable` | `metal::program::{StageExecutable, ProgramStage, ProgramExecutable}` | ported: `compile_program` |
| `M1PreparedFire` | `metal::fire::PreparedFire` | ported: `prepare`/`execute` |
| `M2EncodedRegion`, `M2CommandPlan` | `metal::fused::{EncodedRegion, M2Command}` | ported: the M2 slice |
| `M3EncodedRegion`, `M3StageCommand`, `M3GroupCommand` | `metal::grouped::M3Group` + its private commands | ported: the M3 slice |

Those last five rows are **forward references, not separate entries**. When
this slice landed, each named the slice that would port it and read `missing`;
all four of those slices have since landed, and the rows that own these types
— with the argument for each — are in the `compile_program`, `prepare`/
`execute`, M2 and M3 sections below. They are listed here only because the C++
declares them in this block, and a reader who starts at line 388 of the C++
should be told where they went.

The find of the slice: **`M3GroupLayout::reserved[3]` is load-bearing, on both
sides of the ABI.** The C++ fills the three words through a field literally
named `reserved` with the per-lane binding stride, the parallel-selection row
count, and the per-lane op stride — and the emitted kernels read all three
(`channel_bindings[dispatch_lane * layout->reserved0 + n]`,
`group_position / layout->reserved1`, `dispatch_lane * layout->reserved2`).
Nothing on either side marks them live. The mirror names them
`binding_stride`, `rows_per_lane` and `op_stride` at the same tested offsets.

The lane-table structs are authoritative in `tensor_compiler::plan::lane_table`
and this crate does not build-depend on the compiler, so they are mirrored and
drift-checked the way `status::FAULT_CLASSES` is: a dev-dependency test
compares size and every field offset, and a second holds the exact MSL text of
the three sidecar declarations against the mirror — the compiler's own
preamble comment concedes the MSL copies have "nothing to pin them to".

`Shape` replaces the arithmetic both `prepare` and `prepare_m3_group` walked
by hand: same formula, written twice, no lane index checked anywhere — and
the record array and slot array are contiguous, so `records[lane_count]` does
not fault, it reads channel slots reinterpreted as a lane record. Offsets
here are bounds-checked and the `static_cast<uint32_t>` truncation of
`channel_slot_offset` is a checked conversion.

The three dropped fields: the program-level `grouped_reason` is written
nowhere and read nowhere (the stage-level one of the same name is real and
lands with `compile_program`); `cache_identity` is written once and read
never; `stage_identity` was a heap-allocated `Vec<u8>` of a `u64`'s bytes
whose only job — the collision guard — `pipeline::stage_cache` already does
with the `u64` it stores beside every entry.

## The buffer view — `src/metal/handle.rs`

| C++ | Rust | |
|---|---|---|
| `SlotHandle` | `Handle` | ported |
| `subhandle` | `Handle::slice` | ported |
| `external_handle` | `Handle::over` | ported |
| `SlotHandle::valid` | — | dropped |
| `SlotHandle::offset` | — | dropped |
| `SlotHandle::elastic` | — | dropped |

The first metal-side slice, and the type everything after it stores and binds.
Its tests are in `tests/device_handle.rs` and need a device, including one that
dispatches a kernel through a sliced address to prove the GPU lands where the
host pointer says.

`subhandle` checked nothing: a span past the base was minted rather than
refused, and a default (invalid) base is `nullptr + offset` — UB that in
practice fabricates a handle whose GPU address *is* the offset, which an
argument table binds like any other number. `slice` refuses the first with the
wrap-safe bound every `Region` uses; the second is unrepresentable, because an
invalid `Handle` is not a value of the type and "no handle yet" is
`Option<Handle>`. That is also why `valid()` is dropped.

`offset` was written at every construction and read nowhere on the launch
path; a diagnostic that wants it is one subtraction away. `elastic` is dropped
because it was per-copy state — a flag saying what type the buffer really was —
and the C++'s own `subhandle` demonstrates the failure mode: its designated
initializer names five of the six fields, so a sub-range of an elastic base
would come out ordinary and pass the `bytes <= size` capacity test with no
pages behind it. Elastic-ness here is the `Elastic` type, which a view cannot
mislay. `external_handle` additionally trusted `device_visible()` without
checking it, so a host-fallback ring would bind as GPU address zero; `over`
starts from a real `MTLBuffer` and refuses one the host cannot address.

The ownership flips from borrow to retain: the C++ view is "borrowed; lifetime
owned by RawMetalContext", a contract kept by hand at every copy. A `Handle`
retains its buffer, so the allocation cannot be freed while a view names it —
what retaining does not answer for is exclusivity over a recycled pool buffer,
which is why a handle still belongs beside the owner it was derived from.

## The program compile — `src/metal/program.rs`, `src/metal/runtime.rs`

| C++ | Rust | |
|---|---|---|
| `M1Runtime` (the caches and the counter) | `Runtime` | ported |
| `M1Runtime::create` | `Runtime::new` | ported |
| `compile_program` | `Runtime::compile` | ported |
| `M1RegionExecutable` | `RegionExecutable` | ported |
| `M2FusedRegionExecutable` | `FusedExecutable` | ported |
| `M3GroupedRegionExecutable` | `GroupedExecutable` | ported |
| `M1StageExecutable` | `StageExecutable` | ported |
| `M1ProgramStage` | `ProgramStage` | ported |
| `M1ProgramExecutable` | `ProgramExecutable` | ported |
| `Impl::compile_cached` | `Compiler::compile_sources` | ported |
| the effects fill loop | `channel_effects` + a port fold | ported |
| `kMetalM2MaxFusedChannels` | `MAX_FUSED_CHANNELS` | ported |
| `kMaxRegionsPerStage` / `kMaxRegionsPerProgram` | `MAX_REGIONS_PER_*` | ported |
| `kPrefillOrdinalLimit` | `ORDINAL_BASE` | ported |
| `PTIR_LIBRARY_*` (use) | `LIBRARY_*` mirrors, drift-checked | ported |
| `PsoCompileTransaction` | — | dropped |
| `Impl::remember_negative` | the `compile` wrapper | dropped |
| `compile_faults` / `PIE_METAL_PTIR_TEST_FAIL_COMPILE_ONCE` | — | dropped |

The 718-line function becomes a walk over modules that already exist —
identity, the bounded caches, the stage cache, the emitted-kernel index, the
metadata walk, the shader splice — plus one new compiler primitive:
`Compiler::compile_sources`, the in-memory counterpart of `compile_batch`,
because the launch path's kernels arrive as host-emitted text rather than
files.

`PsoCompileTransaction` is dropped because the thing it rolled back stops
happening: the C++ installs as it builds (PSOs registered with the context,
ordinals taken from the shared counter), so failure needs a destructor that
walks it all back. Here nothing is installed until everything has compiled —
the PSOs sit in a local vector that releases itself, the ordinal counter is
written back in the success path's last statements, and the stage cache is
staged only at assembly, past the last failure exit. A device test proves the
ordinal counter is untouched by a failed compile.

Behaviour changes, each argued in its module: "cache full" is no longer a
retryable failure (the caches evict — `cache.rs`, `stage_cache.rs`); an
in-flight signature collision between two stages of one program builds the
second stage unshared instead of writing a program error into the negative
cache; and the per-region `.mtl4archive` files become one archive per program
(device + combined signature + versions), written by a compiler created for
that build. What that trades away: a new program sharing a stage with an old
one recompiles the stage once after a restart; the in-memory cache still
dedups within a run. A device test shows a second runtime replaying all six
pipelines of a program from the archive.

Two C++ holes are closed in passing: the parallel-top-k classification
indexes `ops[nodes.front()]` unchecked (UB on a malformed plan; a checked
lookup that answers "not the parallel path" here), and the singleton region
cap was checked against the partition's region count while the compile loop
ran one region per *op* — the bound here takes the larger of the two counts.
The channel-effects fold gains the descriptor ports: a consuming port is a
take the descriptor phase performs, which the op walk cannot see, so it is
folded in as a synthetic op list bound by the identity table.

`compile_faults` and its env var are dropped because they existed to reach
caches a test could not otherwise touch; a test here hands `compile` a source
that does not compile. `remember_negative` is the `compile` wrapper's four
lines. `Runtime::new` no longer creates the context — the runtime's state is
caches and counters, and every method takes the `Context` it runs against.

## The single-lane fire — `src/metal/fire.rs`

| C++ | Rust | |
|---|---|---|
| `M1Runtime::prepare` | `Runtime::prepare` | ported |
| `M1Runtime::execute` | `Runtime::execute` | ported |
| `M1PreparedFire` | `PreparedFire` | ported |
| `M1DeviceInputs` | `DeviceInputs` | ported |
| `M1ExecutionMode` | `Mode` | ported |
| `M1PrepareOutcome` | `Prepare` + `readiness::Readiness` | ported |
| `M1ExecuteOutcome` | `StatusOutcome` + `Execution` | ported |
| `Impl::bind_effect_kernel` | the effect-bind loop | ported |
| `M1Runtime::release` | `PreparedFire`'s `Drop` | dropped |
| `resource_accounted` / `m1_prepared_resource_counters` | — | dropped |
| the `goto cleanup_failure` block, twice | `Transient`'s `Drop` | dropped |
| `m1_singleton_fallback_inputs` | — | missing: `batch/` |

The fire takes `Rc<Ring>`s where the C++ took `shared_ptr<ChannelState>`s,
because on the device path the ring's storage must be GPU-addressable and the
Rust host `ChannelState` deliberately is not. `DeviceInputs` drops both
sentinels (`logits` is an `Option<Handle>`, `mtp_draft_row` an `Option<u32>`)
and defers `logits_rows` to the M3 slice, its only reader.

The C++ needed a `goto cleanup_failure` label because four transient buffers
had to be recycled on each of nine failure exits — and the cleanup was still
written twice. Here every `?` is that label: a `Transient` recycles on drop.
`release()` and its `resource_accounted` guard are gone the same way — the
fire's `Rc`s on its rings are what the external-buffer registration and the
global counter were imitating.

The status readback goes through `StatusOutcome::of`, so the two states the
C++ swallowed into "generated op fault 0" — never written, never finished —
keep their names, and a retried fire re-zeroes its status buffer first
(`tests/device_fire.rs` runs one fire twice to hold that). The report prints
the fault class by name via `describe_fault` rather than the code in decimal.

Four device tests run a whole fire — readiness, regions, commit — against
hand-written kernels that honour the real binding ABI and status protocol:
commit lands, a device-side retry reports as retry with its account, an
early fire is refused by the host with zero pool allocations, and a
prepared fire survives re-execution.

## The placed path — `src/metal/fused.rs`

| C++ | Rust | |
|---|---|---|
| `prepare_m2_command` | `Runtime::prepare_m2` | ported |
| `set_m2_logits_row` | `M2Command::set_logits_row` | ported |
| `encode_m2_pre` / `encode_m2_post` | `M2Command::encode_pre` / `encode_post` | ported |
| `finish_m2_command` | `M2Command::finish` | ported |
| `bind_m2_effect` | `M2Command::bind_effect` | ported |
| `bind_m2_region` | `M2Command::encode_region` | ported |
| `M2EncodedRegion` | `EncodedRegion` (private) | ported |
| `M2CommandPlan` | `M2Command` | ported |
| `M2CommandPlan::target` | — | dropped |
| the prepare-time effect binds (2208–2235) | — | dropped |
| — | `M2Command::encoded` | added |

The `RawMetalContext* target` field is dropped: the target's context and
tables are arguments to the calls that need them, so a command cannot be
encoded against one context and finished against another through a stale
pointer. The C++ bound the effect tables twice — once at prepare, again at
every encode — and the encode is when the tables must be current, so the
prepare-time copy is gone.

`encoded` is the lesson the C++ M3 path had learned and its M2 path had not:
a command is prepared before the forward it rides is encoded, and a refused
forward leaves the status zero-filled, which `finish_m2_command` read back
as `"Metal M2 fused execution fault 0"`. `finish` here answers
`NeverDispatched` through the same `Outcome::of` as every other path, and a
device test holds it.

`finish_m2_command`'s hand-ordered teardown — four recycles, every external
release, every ordinal forget — reduces to dropping the command; only the
ordinal forgets stay explicit, because the tables belong to the target.
Three device tests: a placed fire runs inside the target's step (pre →
forward gap → post) and commits with the fused region demonstrably run; a
never-encoded command reports "never encoded"; an unfusable stage is refused
with the host's own reason.

## The group — `src/metal/grouped.rs`

| C++ | Rust | |
|---|---|---|
| `prepare_m3_group` | `Runtime::prepare_m3` | ported |
| `encode_m3_pre` / `encode_m3_post` | `M3Group::encode_pre` / `encode_post` | ported |
| `finish_m3_group` | `M3Group::finish` | ported |
| `bind_m3_effect` / `bind_m3_region` | the encode binds | ported |
| `M3LaneCandidate` | `LaneCandidate` | ported |
| `M3GroupStats` | `GroupStats` | ported |
| `M3EncodedRegion` / `M3StageCommand` / `M3GroupCommand` | private structs / `M3Group` | ported |
| the 64-lane cap | `MAX_LANES` | ported |
| `kM3RegionThreads` | `REGION_THREADS`, drift-checked | ported |
| `M3GroupCommand::target` | — | dropped |
| `timestamp_heap` / `m3_gpu_timestamps_enabled` | — | dropped |
| the `release_group` lambda | ownership | dropped |
| `M1DeviceInputs::logits_rows` | `DeviceInputs::logits_rows` | ported |

The 220-line `release_group` lambda — six group transients, up to seven more
per stage, every external registration and the timestamp heap, recycled by
hand on fourteen failure exits — is ownership here: every early `?` drops
the group and everything it holds. The timestamp heap is dead code by its
own hand (`m3_gpu_timestamps_enabled()` returns `false`; the C++ prices the
feature at 5.0ms of a ~13ms token) and the host-clock fallback it fell back
to is `GroupStats::post_forward_critical_ns`, kept. `kM3RegionThreads` was
the transcription the emitter's own doc warns about — "a hand-kept copy
carrying a 'must equal' comment has nothing comparing the two" — and is now
compared, by a dev-dependency test against `METAL_M3_REGION_THREADS`.

The `encoded` guard — the one lesson this path had already learned — keeps
its shape, and `finish` bounds the fault report at four lanes rather than
letting one bad group log a novel. The `reserved[3]` words of the layout
record are written through their real names (`binding_stride`,
`rows_per_lane`, `op_stride` in `lane.rs`), which is where this ledger's
struct-zoo slice started.

Four device tests, the first of which is the path's whole point: two fires
sharing a stage identity and size bucket become **one** region dispatch
(`body_launches == 1` for two committed lanes). A never-encoded group says
so once instead of faulting every lane; a lane gone stale since composition
aborts the group naming the readiness check; two lanes sharing a ring are
refused as the ordering hazard they are.

## Closed out

Every line of `csrc/src/pipeline/m1_runtime.cpp` (3411 lines, plus its
202-line header and the `region_support.hpp` walkers) is now accounted for
in this ledger: ported with an argued difference, or dropped with the reason
the C++ needed it and the Rust does not. The one entry that remains `missing`
— `m1_singleton_fallback_inputs` in the fire section — is a field copy out of
`batch::MemberForwardDesc`, and belongs to the `batch/` port that owns that
type.

The two entries that used to sit beside it, `m1_extents_from_forward_desc` and
`m3_extents_from_forward_desc`, have since landed there as
`batch::ForwardDesc::extents` and `ForwardDesc::extents_from_readout` — as
methods on the type that owns the fields, which is why they could not be
written here. See `PARITY-BATCH.md`'s `member.rs` section.

## Where this stands

Thirteen subjects ported, each one argued from a specific defect in the C++
rather than from a wish to have it in Rust. The portable half of
`m1_runtime.cpp` — everything that is a function of the plan and the fire's
numbers rather than of the device — is done, and it carries 134 tests that run
without a GPU. The C++ had none for any of it: every one of these functions
lived in an anonymous namespace behind a pimpl, reachable only through a
`*_for_test` hook or not at all.

The metal half is done: the buffer view, the program compile, the device
ring, the single-lane fire, the M2 placement and the M3 group — thirty-two
device tests between them, including end-to-end fires on all three paths
against a real GPU. The port of `m1_runtime.cpp` is complete; what remains
of the driver is the subsystems around it, `batch/` first.
