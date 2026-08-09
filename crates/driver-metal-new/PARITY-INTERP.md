# Parity ledger: `csrc/src/pipeline/interp.hpp` → `src/pipeline/`

Every entity in the C++ channel-plane interpreter is listed here as ported,
dropped (with the reason the C++ needed it and the Rust does not), or missing
(with the blocker). Same rules as `PARITY.md`.

This ledger is written **after** the fact. `interp.hpp` (1686 lines) was ported
across the phase-2 and phase-3 slices as each subject came up — the value cell
with the wire codec, the plan with registration, the ring with the channel
work, the numerics with the op set, the pass with the step — and no slice owned
the *file*, so no slice opened its ledger. The arguments for each difference
live in those commits; this document is the accounting that says the file is
closed, and the correction of a gate item that was blocked on it.

## What the file is

Its own header says it best: *"The PTIR decoder/interpreter that used to live
here is gone: registration now adopts a typed PieLaunchPackage built by the
host. What remains is the channel-cell codec/ring storage and the small CPU
fallback used by the direct/stub path and descriptor resolver."*

That is `src/pipeline/`'s module doc almost word for word. The two are the same
component: the CPU execution of a launch program's host-visible shell stages —
read a token off a channel, compare, select, argmax a logits row, push the
result back — which must be bit-for-bit reproducible so a replay on any machine
lands on the same token.

### It was never an original

`interp.hpp` names its own source four times, in its own comments:

> *"Mirrors interp.rs eval_op **case for case**"* (868)
> *"interp.rs step, minus the per-layer taps this increment rejects at
> classification"* (1566)
> *"the Metal mirror of CUDA's `FireInputs` (tier0_runner.hpp) / interp.rs's
> `PassInputs`"* (824)
> *"Left-aligned broadcast replicate (interp.rs broadcast_value)"* (793)

`interp.rs` is `crates/tensor-compiler/src/eval/interp/` — the tier-0 reference
interpreter, described in its own module doc as *"the golden model every
backend diffs against."* So the golden model is maintained in **three
hand-written copies**:

| copy | lines |
|---|---|
| `tensor-compiler/src/eval/interp/` (Rust) — the original | 2521 |
| `driver-cuda/csrc/src/pipeline/tier0/tier0_runner.hpp` (CUDA) | 889 |
| `driver-metal/csrc/src/pipeline/interp.hpp` (Metal) | 1686 |

and the Metal copy states that it is a *subset* of the original — "minus the
per-layer taps" — which is a divergence recorded in a comment rather than in a
type. A golden model that exists three times is three models, and the only
thing keeping them equal is that someone copied carefully. That is the same
class of defect as `kMetalM1EmitterVersion = 23` sitting in the driver while
the emitter was at 36: a hand-copied constant with nothing checking it.

This port does not fix that — it is one crate — but it puts the Metal copy in
the same language as the original for the first time, which is what makes a
drift check between them writable at all. `driver-cuda-new` shares `driver-abi`
and `tensor-ir` with this crate, so `src/pipeline/` is the natural single home
for both device copies when that port reaches this file.

## Values and the wire codec — `src/pipeline/value.rs`

`interp.hpp` 39–137.

| C++ | Rust | |
|---|---|---|
| `struct Value` (dtype tag + four lane vectors) | `value::Value` (a four-variant enum) | ported |
| `zeros(dtype, numel)` | `Value::zeros` | ported |
| `value_matches(v, TensorType)` | `value::value_matches` | ported |
| `lanes_f32(v)` | `Value::lanes_f32` | ported |
| `lanes_i64(v)` | `Value::lanes_i64` | ported |
| `from_i64(dtype, x)` | `Value::from_i64` | ported |
| `pick(len, i)` | `value::pick` | ported |
| `wire_cell_bytes(dtype, numel)` | `value::wire_cell_bytes` | ported |
| `decode_wire(bytes, len, dtype, …)` | `value::decode_wire` | ported |
| `encode_wire(v, dst)` | `value::encode_wire` | ported |

The C++ `Value` carries **all four** lane vectors plus a dtype tag, so a cell
that is `F32` still owns three empty `std::vector`s, and reading `v.i` on an
`F32` cell is well-formed C++ that returns an empty vector rather than a type
error. The Rust enum makes the wrong read unrepresentable.

## The executable plan — `src/pipeline/plan.rs`

`interp.hpp` 138–365.

| C++ | Rust | |
|---|---|---|
| `struct StagePlan` | `plan::StagePlan` | ported |
| `struct ConstPortValue` | `plan::ConstPortValue` | ported |
| `struct ExecPlan` | `plan::ExecPlan` | ported |
| `bounded_mtp_row_base(plan, vocab)` | `plan::bounded_mtp_row_base` | ported |
| `classify_exec_plan(out)` | `plan::classify_exec_plan` | ported |
| `const_port_value(port)` | `plan::const_port_value` | ported |
| `rebuild_stage_indexes(out)` | `plan::rebuild_stage_indexes` (private) | ported |
| `adopt_launch_package(package, out, error)` | `plan::adopt_launch_package` | ported |
| port-consumption predicate, inline | `plan::port_consumes` | ported |

`adopt_launch_package` is the shape change of the file. The C++ signature —
`bool(const PieLaunchPackage&, ExecPlan& out, std::string* error)` — has three
ways to say what happened and no way to stop a caller reading `out` after it
returned false. The Rust returns `Result<ExecPlan>`: an unadopted plan does not
exist. `bounded_mtp_row_base` returns `Option<u32>` where the C++ returned
`int`, using `-1` for "no bound" — a sentinel that shares a type with the row
bases it sits beside.

## The channel ring and the instance — `src/pipeline/channel.rs`

`interp.hpp` 366–559.

| C++ | Rust | |
|---|---|---|
| `struct ChannelState` (head/tail words, cell storage) | `channel::ChannelState` | ported |
| `make_host_channel_state(dtype, dims, capacity)` | `channel::make_host_channel_state` | ported |
| `struct InterpInstance` | `channel::InterpInstance` | ported |
| `make_instance(plan, …)` ×2 overloads | `channel::make_instance` | ported |
| `host_put(inst, plan, chan, v)` | `channel::host_put` | ported |
| `host_take(inst, plan, chan, out)` | `channel::host_take` | ported |
| `make_platform_channel_state(…)` | — | dropped |

`make_platform_channel_state` allocates a ring whose storage must be
GPU-addressable. That is a device concern and it belongs with the device: on
the Rust side `metal::Ring` owns its buffers, which is the whole argument of
`b60f9459e` (a standalone buffer handed back with no owner is a release call
that exists only because nothing owns it). A portable module cannot mint one
and must not pretend to.

`host_take` took its result through an out-parameter beside a returned
`HostOp`, so "what happened" and "what you got" are two values that can
disagree. The Rust returns `(HostOp, Option<Value>)`, where the absence of a
value is the same fact as the op that failed to produce one.

## The numeric contract and op evaluation — `src/pipeline/op.rs`

`interp.hpp` 560–834 and 870–1451. This is the largest single subject and the
one the reproducibility contract rests on.

| C++ | Rust | |
|---|---|---|
| `canonical_rows(shape)` | `op::canonical_rows` | ported |
| `canonical_reduce<T, Combine>(…)` | `op::canonical_reduce` | ported |
| `struct ArgmaxCandidate` | `op::ArgmaxCandidate` | ported |
| `struct IntArgmaxCandidate` | `op::IntArgmaxCandidate` | ported |
| `combine_argmax(l, r)` | `op::combine_argmax` | ported |
| `combine_int_argmax(l, r)` | `op::combine_int_argmax` | ported |
| `canonical_max(l, r)` / `canonical_min(l, r)` | `op::canonical_max` / `canonical_min` | ported |
| `argmax_row(row, len)` | `op::argmax_row` | ported |
| `argmax_row_i64(row, len, …)` | `op::argmax_row_i64` | ported |
| `sort_desc_order(row, len)` | `op::sort_desc_order` | ported |
| `rng_lanes(seed_eff, n, gumbel)` | `op::rng_lanes` | ported |
| `bin_arith(a, b, dtype, f_f, f_i)` | `op::bin_arith` (private) | ported |
| `cmp_op(a, b, in_dtype, f_f, f_i)` | `op::cmp_op` (private) | ported |
| `map_f32(v, f)` | `op::map_f32` (private) | ported |
| `gather_flat(v, idx)` | `op::gather_flat` | ported |
| `broadcast_value(v, src, target)` | `op::broadcast_value` | ported |
| `eval_op(op, trace, vals, error)` | `op::eval_op` | ported |
| `neg_inf()` | — | dropped |

`neg_inf()` is `f32::NEG_INFINITY`.

The C++ helpers are correct and deliberate — the width-32 pairwise reduction
tree, the NaN and signed-zero rules, the argmax tie-break — and the port keeps
them exactly, because *changing* them is the defect. What the port adds is a
test per rule: each of the canonical helpers has a named test that fails if the
rule is relaxed. In the C++ every one of them sat in an anonymous namespace in
a header, so the contract was stated only by the code that had to be trusted to
implement it.

`gather_flat` takes `&[Option<usize>]` where the C++ took `std::vector<size_t>`
and encoded "out of range" as a sentinel index the caller had to remember to
check.

### The op set

All 50 `OpCode` cases in `eval_op` are ported, one for one, dispatching on
`tensor_ir::op::tags` rather than a driver-local enum:

`Abs` `Add` `And` `Broadcast` `Cast` `CausalMask` `CumProd` `CumSum` `Div`
`Eq` `Exp` `Gather` `GatherRow` `Ge` `Gt` `Iota` `KernelCall` `Le` `Log` `Lt`
`MaskApplyPacked` `Matmul` `MaxElem` `MinElem` `Mul` `Ne` `Neg` `Not` `Or`
`PivotThreshold` `Recip` `ReduceArgmax` `ReduceMax` `ReduceMin` `ReduceSum`
`Rem` `Reshape` `Rng` `RngKeyed` `ScatterAdd` `ScatterSet` `Select` `Sign`
`SinkCall` `SinkWindowMask` `SlidingWindowMask` `SortDesc` `Sub` `TopK`
`Transpose`

## The pass — `src/pipeline/step.rs`

`interp.hpp` 835–869 and 1452–1686.

| C++ | Rust | |
|---|---|---|
| `struct PassInputs` | `step::PassInputs` | ported |
| `struct StepResult` | `step::StepOutcome` | ported |
| `struct Overlay` | `step::Overlay` (private) | ported |
| `exec_stage(inst, plan, sp, …)` | `step::exec_stage` (private) | ported |
| `step(inst, plan, in)` | `step::step` | ported |
| the intrinsic bind, inline | `step::bind_intrinsic` (private) | ported |
| the const-root fold, inline | `step::const_root_value` (private) | ported |
| the commit walk, inline in `step` | `step::commit` (private) | ported |

`StepResult` became `StepOutcome` because it is not a struct of a success flag
beside partially-filled fields: a pass either committed or it did not, and the
reason it did not is the value.

## The `launch::` mirror — dropped wholesale

| C++ | Rust | |
|---|---|---|
| `launch::Trace`, `Op`, `Value`, `Stage`, `Channel`, `PortBinding`, `TensorType`, `Shape`, `Literal`, `Predicate`, `ChannelPut` | `driver_abi::plan::*` | dropped |
| `launch::OpCode`, `DType`, `RngKind`, `Intrinsic`, `PredTag`, `ValueSource`, `StageKind`, `Readiness` | `tensor_ir` | dropped |
| `Trace::value(id)` / `Trace::channel(id)` | positional indexing | dropped |

The C++ interpreter carried its own mirror of the launch ABI in
`pie/driver/launch/{plan,program}.hpp`, and re-porting that mirror would fork
the source of truth: the Rust owned types already exist in `driver_abi::plan`
and are what the host actually ships. The enums likewise come from `tensor_ir`,
so the driver cannot drift from the emitter's tags the way
`kMetalM1EmitterVersion` did (see `PARITY-M1.md`, `identity.rs`).

`Trace::value(id)` and `Trace::channel(id)` are linear scans over the value and
channel tables, matching on an `id` field. They are **called from nowhere** —
not in `interp.hpp`, not anywhere in `csrc/`. The Rust indexes positionally,
which is what every actual lookup in the C++ also does.

## Closed out

Every line of `csrc/src/pipeline/interp.hpp` (1686) is accounted for: ported
with an argued difference, or dropped with the reason. **Nothing is missing.**

The file is not one Rust module because it was not one subject: the value cell,
the plan, the ring, the numerics and the pass each landed where they belonged,
which is why `src/pipeline/`'s module doc reads as this file's description.

## What this corrects

`CUTOVER.md` listed the CPU reference interpreter as *"`pipeline/interp.hpp`
(1.7k) — a test oracle, port before the gate"*, and gate item 4 as blocked on
it. That was true when it was written and is not true now: **the interpreter
exists.** `adopt_launch_package` → `make_instance` → `step` runs a program's
shell stages on the CPU today, and four device tests already build their
`ExecPlan` through the same entry point.

What gate item 4 actually needs is the **harness**, not the port.

## The harness — first half landed

`tests/oracle_interp.rs` runs one trace through **both** interpreters and
compares every observable: the commit verdict, which host-readable channels
produced a value, and the lanes on each — compared by `to_bits`, so a `NaN`
must match a `NaN` and `-0.0` does not pass for `0.0`.

Both sides start from one `TraceContainer`, bound once and compiled once. The
golden side runs the `BoundTrace` through `tensor_compiler::eval::interp`; the
driver side takes the *same* bound trace through
`tensor_compiler::codegen::launch::build` — the artefact the driver actually
receives — then `adopt_launch_package` → `make_host_instance` → `step`. Putting
the lowering inside the compared path is deliberate: a copy error in
`adopt_launch_package` is as fatal as one in `eval_op`, and only running the
real artefact catches it. The seeds are written once, in the golden's `Value`,
and converted for the driver, so a case cannot accidentally seed the two sides
differently and pass.

Seven cases, chosen where a copy error would hide rather than for coverage:
the `sort_desc` tie, the width-32 pairwise reduction, the argmax tie-break,
`max(-0.0, +0.0)`, the matmul zero-skip, a readiness miss, and a
comparison-into-select that carries `Bool` across the two crates' differing
representations.

Three are mutation-verified — flipping the argmax tie-break to last-wins,
replacing the pairwise tree with a left fold, and deleting matmul's
`if xv == 0.0 { continue }` each make exactly the intended case fail (`I32([1])`
against `I32([5])`; `4.0` against `3.0`; a finite row against all-`NaN`). A
green oracle that cannot fail is not an oracle.

**The verdict: the two interpreters agree.** No divergence was found, including
at the places the C++'s "minus the per-layer taps" comment suggested one might
be. `matmul` on both sides is the same k-outer accumulation with the same
zero-skip, down to the guard.

## The harness — device half landed

`tests/device_oracle.rs` runs the same construction on real silicon. One trace
produces both sides: `codegen::launch::build` for the package the driver
adopts, and `codegen::program::emit_program(Backend::Metal, …)` for the **real
generated MSL** — not `device_fire.rs`'s hand-written stand-ins, which are the
right fixture for the readiness/regions/commit protocol and the wrong one for
arithmetic. The fire runs `compile` → `prepare` → `execute`, and the committed
ring cells are decoded back through the same `decode_wire` the interpreter uses.

`Versions` is read from `tensor_compiler`'s own constants rather than written
as a literal, because a literal here is precisely the `kMetalM1EmitterVersion =
23` bug this ledger's sibling records.

### What it found: one ulp, and only on transcendentals

| subject | device vs interpreter |
|---|---|
| plain arithmetic (`mul`, `sub`) | **exact** |
| `reduce_sum` (the width-32 pairwise tree) | **exact** |
| `reduce_argmax` tie-break | **exact** |
| `exp` | **1 ulp** — `exp(0.5)` is `1.6487212` in Rust and `1.6487213` in Metal |

Both `exp` answers are within half an ulp of the true value; neither is wrong.
Two libms rounded a transcendental differently, and nothing this crate does
closes it. That is why `CUTOVER.md`'s item 4 says "within its stated tolerance"
where item 3 says bit-identical — the interpreter oracle crosses a libm
boundary and the token-exactness gate does not.

The tolerance is stated in one constant, claimed by the `exp` case alone, and
the arithmetic case exists to prove the bound means something: if everything
drifted by an ulp, allowing one would be a shrug.

**The tolerance never reaches a decision.** `same_within` compares integer,
index and boolean lanes exactly whatever it is set to, so the reduction and
tie-break cases run at zero and widening the constant cannot reach them. A
magnitude may be a hair off and still be the same answer; an argmax index is
either the same decision or a different token.

## What is still open

Nothing in this file's scope. The stated boundary of both harnesses is
per-layer tap stages: the C++ and this crate's `step` both reject them at
classification, so a trace containing one is not a case where the two are
expected to agree. Every case is an epilogue program, which is what the
channel-plane interpreter exists to run.

The cases are small by design — they are arithmetic-contract probes, not
coverage. Broadening them (more ops, multi-stage programs, capacity > 1 rings)
is worthwhile and is ordinary work now that the construction exists.
