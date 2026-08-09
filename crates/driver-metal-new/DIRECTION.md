# Direction: the model-compiler path, and what it retires

> The planning map above this document is `.wiki/new-driver/metal.md`: the
> north star in full (general-purpose executor, tart-level supergraph
> polymorphism, no hardwired model definitions, end-to-end function), where
> Metal stands against each, and the remaining tasks in dependency order. This
> file stays the decision record for the ones already taken.

Decided 2026-08-10. **Read this before picking work out of `PARITY-BATCH.md`
or `CUTOVER.md`** — both were written against the older plan and describe work
that is no longer worth doing.

## The north star

`crates/model-compiler/DSL-DESIGN.md` states it in one line:

> **Nothing in the driver may choose a kernel.** A statement names the symbol
> it runs; the driver resolves weight names to pointers, resolves value ids to
> addresses, and calls. That is the whole of its job.

A traced fire is lowered by `model_compiler::lower` into a flat list of
`Launch` rectangles, each naming a kernel symbol and carrying its operands as
`Arg`s. The executor binds and dispatches. There is no per-family forward.

**Metal is going all in on this**, alongside CUDA.

## This is a seam that already exists, not a new architecture

`model-compiler` already depends on **both** kernel tables and already has the
backend it needs:

```
crates/model-compiler/Cargo.toml
    kernels       = { path = "../kernels" }
    kernels-cuda  = { path = "../kernels-cuda",  default-features = false }
    kernels-metal = { path = "../kernels-metal", default-features = false }

crates/model-compiler/src/kernels.rs
    pub enum Backend { Cuda, Metal }
    Backend::Metal => KERNELS_METAL
```

## Three legs, and only one of them is done

Going all in needs three things. **All three are now done as code**, and what
is left is the proof: no run against a real checkpoint has been held to
`tests/device_smoke.rs` token for token, and until one has, this is a path that
executes rather than a model that answers.

### 1. The lowering — **done**

`Lowered` is backend-neutral by construction: `launches`, `kernels:
Vec<String>` (symbols, not function pointers), `arena_bytes`, `value_offset`.
Nothing in it is CUDA-shaped, and `Backend::Metal` resolves to `KERNELS_METAL`
today.

### 2. The Metal DSL text — **one family, and it now states an axis**

`model-compiler` compiles a DSL, and **a text has to be written for Metal**:
`dsl::trace_metal(family, ..)` records `<family>.metal.<class>`, and the
symbols the body names must be Metal symbols. A CUDA text does not serve.

What exists: `crates/model/src/families/llama_like/forward/mod.rs`'s
`llama_like_metal_text`. Its own doc states the gaps, and they are the work:

* ~~the M>1 lane is a guess~~ — **checked and closed, 2026-08-10.** The live
  `MbFeatures` for this family is `{ gdn, sdpa_d256 }`: split-k, fp16-precast,
  strided, fp16-strided, d512 and residual are `false` in every live path and
  true only in a test fixture, and `PARITY-BATCH.md` already records them as
  deferred with reasons. `bias` belongs to gpt-oss and `routed` to a mixture
  `llama_like` does not model. The driver carries rungs nothing turns on; the
  text states the lane that fires. Only the `kQmmMinBatch` gate is untested,
  and the text takes it as a load-time fact rather than deciding it;
* ~~`sdpa_*_d_256` pins head_dim 256~~ — **fixed 2026-08-10**, and it was a
  defect, not a simplification: `qwen3_0_6b`'s heads are 128 wide and the text
  named a 256-wide kernel, which reads past the end of every head without
  faulting. `dsl::metal::sdpa` takes `head_dim` now;
* **no seams** — the adapter, the two observation taps and the boundaries the
  CUDA text states are absent, "because none of the machinery behind them
  exists on this backend yet";
* qk-norm and bias are stated as ordinary norms and are **untested** against
  what `declared_dag.hpp` expects — though they are now *asked of the tensors*
  rather than assumed (`model::text::facts_from`);
* ~~the text is monomorphic~~ — **closed 2026-08-10.** `m.depth_window()`
  makes every layer-tagged statement implicitly `rows(depth > layer)`. On eight
  prefill rows with half truncated at layer 4, row-work falls **2936 → 1688
  (−42%) at the same 367 launches**: the shared prefix executes once, which is
  the supergraph claim. Stating an axis also buys the **seriation contract** —
  the text now refuses `Discontiguous { axis: "depth" }` on a row order whose
  depth runs are not contiguous, so the frame bridge must hand rows over
  seriated;
* **four defects of one shape, all found by making it run.** The text assumed
  bindings the checkpoint does not have, and not one would have failed loudly:
  affine projections named one tensor where the kernel reads three; symbols
  named STEMS where shaders export instantiated points; the gated MLP passed
  one packed value where `silu_mul` takes gate and up as two buffers, so the
  OUTPUT bound where `up` belongs; and the seam took a test fixture's facts for
  every checkpoint. The cure is the same each time — **ask the tensors**. A
  config states an architecture; a tensor states a binding.

What does not exist: a Metal text for any other family. `crates/model/src/
families/` holds only `llama_like`, while the Metal driver carries handwritten
forwards for llama, gemma4, gpt-oss and qwen. **Every one of those needs a text
before its handwritten forward can go.**

Related: `kernels-metal` has **98** `kernel!` rows against `kernels-cuda`'s
**226**. A symbol a text wants to name needs a row, because the row is where
the contract lives.

(`trace_metal`'s doc comment still says "nothing calls it, and the empty Metal
kernel table". Both were true when written and neither is now — one caller, 98
rows. Same staleness this crate's ledgers keep showing.)

### 3. The Metal executor — **done**

`src/model/` is it, and the shape is the argument:

| | |
|---|---|
| `executor.rs` | binding — three resolution rules, stated once |
| `geometry.rs` + `grid.rs` | a rectangle's thread grid, by the rule its row names |
| `dispatch.rs` | the walk: symbol → row → file → rule → grid → operands |
| `frame.rs` | a sealed frame's step → `&[Row]` |
| `kv.rs` | the paged pool, sized by the fire |
| `load.rs` / `resolve.rs` / `text.rs` | the checkpoint, the name map, which text |
| `encode.rs` / `run.rs` | the device half: compile by name, bind, dispatch |

**There is no arm per kernel and no branch per family.** `dispatch::plan_one`
is: look the symbol up, read its file, evaluate its rule, bind its operands.
`tests/device_text_fire.rs` runs all 367 launches of `llama_like`'s text on the
GPU, in both fire classes, through that walk.

Two things it proved on the way, both worth keeping:

* **the M>1 lane was never a second vocabulary.** The plan recorded "which of
  the two rule sets a row means" as a question to answer first. Measured, every
  M=1 function is its M>1 function at ONE ROW, and where the lanes genuinely
  differ they are *different kernels with different names*. The lane is
  `dims.rows`;
* **the frame bridge had a predecessor, and it was not the C++.** tart rung
  ③'s region table maps bit-for-bit onto `Row` — the scheduler already computes
  the seriation, so the driver reads it rather than deriving it.

And the seam: `engine/src/driver/backend/metal.rs` is a **plain Rust library
call**, not an ABI crossing, because the driver on the other side is Rust. That
is task 9 arriving from the other end — nothing here adds a C boundary for it
to remove.

**Thirteen of its fourteen verbs are served.** `encode` refuses because Metal
media encode is unsupported on this backend and on CUDA both. `resize_pool`
and `copy_state` refuse by name: a resize means reallocating a pool that is a
fixed allocation today, and `copy_state` wants recurrent-state slots no
`llama_like` deployment has. Neither blocks serving a dense model.

Two of those verbs were left refusing on reasons that did not survive looking,
and both are worth recording because the rule that found them is the crate's
own — *before starting anything a ledger calls missing, look for it*:

* the **registry three** were said to need the ring's device addresses.
  `ChannelState` holds `RefCell<Vec<u8>>` and four `AtomicU64`s: the channel
  plane is host memory here exactly as it is on the dummy driver, and the
  binding is those two addresses. They gate everything — without them no
  instance is bound, so no `FrameSubmission` is built;
* **`copy_kv`** was said to need an encoder. The pool is `StorageModeShared`,
  so a move is a `memmove` — and `Region::copy`'s memmove semantics are the
  point, because a compaction slides rows and the spans overlap.

### Metal is not behind on kernel resolution — it is ahead

Worth stating because it is easy to assume otherwise. Both kernel crates are
the same shape: a `KERNELS` table of `KernelSig` rows built on the shared
`kernels` crate. The difference is how a symbol is reached:

* **Metal** resolves by **name string at runtime** — `Compiler::compile(context,
  source, function: &str)` builds a pipeline state from an entry-point name.
  A symbol the lowering states can be reached without the driver having been
  written to know it exists.
* **CUDA** reaches `pie_k_*` C symbols through a dispatch arm per kernel, which
  `executor.rs` says "grows kernel by kernel beside the bridge".

So the mechanism the north star needs is already in place on Metal. What is not
in place is that **the plans deciding which symbols to use are written per
family, by hand** — `psos_llama.rs`, `psos_gemma4.rs`, `psos_gptoss.rs`,
`psos_mb.rs`. Those are the driver choosing kernels, and they are what the
lowering replaces.

## The C++ driver is retired as of 2026-08-10

Executed, not planned. `crates/driver-metal` is out of the workspace members
and out of `engine`'s dependency graph; `engine/src/driver/backend/metal.rs`
and the `DriverBackend::Metal` variant are deleted, and the `driver-metal`
feature is gone from `engine`, `worker`, `tests/gpu` and the root manifest.
The source stays as reference — `crates/driver-metal/README.md` says what is
still worth reading in it and why.

**The consequence, stated plainly: Metal has no serving backend right now.**
The C++ was already off by default (`worker`'s default feature set is empty),
so no default build changes, but the option is gone and nothing replaces it
until the executor below lands. That is the deliberate order — the alternative
was maintaining a driver whose shape is retired while building the one that
replaces it.

## What this retires

Roughly 8.5k lines across 21 files, plus the qwen path embedded in the shared
modules:

| retired | what it is |
|---|---|
| `batch/dispatch_{llama,gemma4,gptoss}.rs`, `dispatch_mb.rs` | per-family DAG builders — the handwritten forward |
| `batch/psos_{llama,gemma4,gptoss,mb}.rs`, `psos.rs` | per-family PSO plans — the driver choosing kernels |
| `batch/{llama,gemma4,gptoss}.rs`, `*_consts.rs`, `gptoss_solve.rs` | per-family geometry and constant walks |
| `metal/{llama,gemma4,gptoss}_{bind,step,engine}.rs`, `step.rs`, `step_mb.rs`, `bind.rs`, `bind_mb.rs` | per-family binds, steps and engines |
| **`forward.cpp` / `forward.hpp` (5393)** | **do not port it.** It is the family executor. It is replaced, not translated |

`PARITY-BATCH.md`'s remaining rows are almost entirely this executor and its
dependents. Those rows are now *obsolete rather than outstanding*, and the
ledger should be read with that in mind until it is rewritten.

## What survives, and it is most of the crate

Everything that is not a family:

| survives | why |
|---|---|
| all of `src/metal/` except the family files — context, device, heaps, pools, elastic, keepalive, encoder/stepper, pipeline compiler, archives, tables, timestamps, timing, residency, handle, ring, fire, fused, grouped, storage, paging | the substrate any executor needs. The lowering names symbols; this is what runs them |
| all of `src/pipeline/` | the PTIR channel-plane interpreter. A **different layer** from the model forward — prologue/epilogue shell stages, channels, readiness, the fire's plan. It is already model-agnostic and is not affected |
| `src/loader/`, `src/store/` | weights, KV pages, recurrent slots |
| `batch/` minus the family files — `schedule`, `mask`, `admit`, `member`, `marshal`, `sequence`, `paged_state`, `tickets`, `color`, `sizing`, `heap_budget`, `fit`, `logits`, `golden`, `timing`, `paging`, `fire_csr`, `abi` | the frame and fleet layer: who is in this fire, which pages, which slots, what fits. The lowering does not answer any of these |
| `src/facts.rs`, `shader.rs`, `tuning.rs`, `region.rs`, `bump.rs` | host-portable substrate |

The work of the last two days is in the surviving column. It was ported for the
C++'s reasons and it holds for the new ones, because none of it chooses a
kernel.

## Two things a symbol does not yet carry

Found while starting the dispatch half, and they are the difference between
"a symbol is a name" being nearly true and being true. Both are questions for
the tables, not for the driver — a driver that answers either is a driver
choosing.

### 1. Which shader file defines it

`Compiler::compile` takes a **source path and an entry name**, so a symbol
alone cannot be compiled. Today the only things that know the file are the
hand-written per-family PSO plans, which carry it as a literal
(`Request::new(kernels_dir.join(r.file), r.entry)`) — one more fact the driver
knows and the statement does not, and one that retires with them.

Nothing else has it. `KernelSig` has no file field; the row carries `name`,
`symbol`, the axes and the contract. `entrypoints.generated.txt` is 479 bare
names. Of the 98 `kernel!` rows, **16 mention a `.metal` file in a comment**,
in prose (`// 7 in sdpa_paged.metal.`) — enough to read, not enough to compile
from, and the usual shape of a fact that lives in a comment.

The fix belongs on the row: a `file` beside the `symbol`, filled per row the
way the signature was. 98 rows of ordinary work, and it cannot be scripted
from the comments.

Note this is a *Metal* problem only. CUDA reaches a `pie_k_*` symbol through a
linked C function, so nothing has to say where it lives — the linker knows.
Metal's runtime compilation is the reason the name works at all and the reason
the file has to be stated.

### 2. Its launch geometry

A row states the contract — `whole`, `needs`, `lacks`, `sink`, the in-place
pairs, the operand sources — and **no grid or threadgroup size**. The
per-family plans carry those too, per kernel and per shape.

A rectangle gives rows and layers, which is the *iteration space*, not the
launch. Something has to turn one into the other, and if it is a `match` on
the symbol in the driver then the arm-per-kernel that Metal was supposed to
avoid has reappeared under a different name.

**Decided, 2026-08-10: the row names a rule, and the rule stays a function.**
`src/model/geometry.rs` is the vocabulary — eleven variants covering the
sixteen geometry functions `batch/dispatch.rs` already hand-writes, with a
test that every one is reproduced exactly.

Why not the alternatives. Numbers on the row cannot work: a geometry is a
function of the fire, so a row would state a formula rather than a value. A
`const` expression grammar *can* express all sixteen — they are uniformly
`source → max → min → divide-rounding-up → multiply` — but writing
`Term { floor: 1, cap: 1024, div_ceil: 32, mul: 32 }` loses the sentence that
says why, and in this codebase those sentences are load-bearing: `qmv`'s doc
records that its round-up is the difference between computing every output and
silently dropping the last few. Reading it off the compiled pipeline gives a
*legal* threadgroup, not the intended one — which is wrong for any kernel whose
algorithm assumes its shape, and those are most of them.

What the decision buys: the executor's dispatch is `sig.launch.eval(dims)` in a
loop, the match is arm-per-RULE rather than arm-per-kernel (eleven arms shared
by every family, text and backend), a kernel that launches like an existing one
costs zero arms, and every doc comment stays beside the arithmetic it explains.

The vocabulary lives in the driver until the tables adopt it, because it had to
be shown to cover the existing rules first. Moving it is mechanical: the row
gains `launch = Rule::Qmv` beside `whole`, `needs` and `lacks`, and
`Rule::Unstated` — the default — refuses, so a symbol whose contract has not
said cannot be dispatched from a guess.

## The next step

The three legs can go in parallel, and the order that de-risks fastest is:

1. **The executor**, against `driver-cuda-new/src/model/executor.rs`. It is
   the smallest of the three and it makes the other two testable end to end.
   Metal's dispatch half should be **shorter than CUDA's**, because a symbol is
   a name here: where CUDA grows an arm per kernel, Metal can compile the entry
   point the lowering named and bind operands in the row's stated order. If
   that holds it is the argument for the whole approach, and it is worth
   proving before the texts are written against it.
2. **Close `llama_like`'s gaps**, in the order its doc lists them. The M>1 lane
   is the one that decides whether the text can replace the driver's
   `MultiBatchPsos` or only its decode step.
3. **A text per remaining family** — gemma4, gpt-oss, qwen — each retiring its
   handwritten forward as it lands, with the device smokes already in
   `tests/device_smoke.rs` as the equality check. Those smokes are
   token-exact against mlx_lm today, so a text is right when it reproduces
   them.

Add `model-compiler` as a dependency when step 1 starts.
