# Loader v2 — North Star Plan

> **Status: proposed, nothing landed.** This is the target state for a full
> rewrite of `loader/` (crate `pie-loader`), driven by three concerns and no
> others: **extensibility** for primitives and passes we do not have yet,
> **removal of abstraction that earns nothing**, and **structural smells**.
>
> It is a plan, not a spec. The declaration format stays specified in
> [`spec.md`](spec.md); the boundary and its history stay in
> [`architecture.md`](architecture.md). This document says what the code should
> look like when those two are unchanged in intent but the implementation has
> been rebuilt beneath them.
>
> Citations of the form `path:line` refer to the **current** tree and exist to
> ground each claim in evidence. Every number in §2 was measured, not
> estimated; §2.4 says how.

---

## 1. The thesis

> **The loader has two IRs: the contract and the plan. The only computation
> between them is solving an affine algebra into rectangular byte copies.**

Everything else is either a view of one of those two, or a pass over the plan.
That sentence is the whole design, and the current tree violates it in one
specific way: it has **four** IRs where it needs two.

| | IR | Lines | Under v2 |
| --- | --- | ---: | --- |
| 1 | `contract::Expr` — the declaration algebra | `contract.rs` 1,329 | **stays** — this is IR #1 |
| 2 | `contract::compile::Lowering` — solved runs and leaves | `contract/compile.rs` 1,746 | **stays** — the solver's result, not a separate IR |
| 3 | `ir::LayoutExpr` — a second algebra | `ir.rs` 175 + `optimizer.rs` 276 + `typecheck.rs` 403 | **deleted** |
| 4 | `load_plan::StorageInstr` — the executable plan | `load_plan.rs` 536 | **stays** — this is IR #2 |

Plus two POD mirrors of #1 and #4 in `ffi/` (4,521 lines including tests).

IR #3 is a vestige of a retirement already in progress. `architecture.md:1459`
records that `LayoutExpr` has already lost eight variants — `ByteSpans`,
`Select`, `Partition`, `Join`, `Stack`, `Unzip`, `Reorder`, `View`. Its own
optimizer says so in the first paragraph of the file (`optimizer.rs:3-14`):

> *"There used to be a great deal more here: pushing selects through joins,
> cancelling a partition against the join beneath it... Every one of those rules
> existed to undo structure the frontend had just built. Now the frontend builds
> none: `contract::compile` resolves the whole affine fragment symbolically
> before an IR node is ever pushed."*

The file is arguing for its own deletion. v2 accepts the argument.

---

## 2. Evidence

### 2.1 Extensibility: the edit surface is the problem

Adding one primitive touches this many files today:

| To add… | Files | Where |
| --- | ---: | --- |
| a `contract::Expr` variant | **8** | `contract.rs` (enum + `infer` + `specialize`) · `contract/compile.rs` (`build`, `step`) · `frontend.rs` · `ffi/types.rs` · `ffi/contract.rs` · `contract_writer.rs` · `reference.rs` · C++ `contract.hpp` |
| an `ir::LayoutExpr` variant | **6** | `ir.rs` (+ `inputs()` + `decl()`) · `typecheck.rs` · `optimizer.rs` · `planner.rs` · `frontend.rs` · `reference.rs` |
| a `StorageInstr` variant | **10** | `load_plan.rs` · `planner/passes.rs` (39 match sites) · `planner/memory.rs` · `planner/arena.rs` · `host_executor.rs` (25 match sites) · `dump.rs` · `ffi/arena.rs` · `ffi/types.rs` · `verify.rs` · C++ executor |
| a plan field | **5** | core struct · `ffi/types.rs` mirror · `ffi/arena.rs` writer · `ffi/view.rs` reader · `pie_loader.h` |
| a pass | **1 + order** | `planner.rs:135-146`, a hardcoded call sequence with no registry |

Roughly half of each of those counts is IR #3 and its POD mirror. Deleting the
middle algebra is the single highest-leverage extensibility change available.

### 2.2 Abstraction that earns nothing

- **`ir::LayoutExpr`'s type checker restates the contract's.** `typecheck.rs`
  (403 lines) re-derives shapes and encodings that `contract.rs::infer` already
  proved. It runs **three times** per compile — `optimizer.rs:48`,
  `optimizer.rs:89`, `planner.rs:55`.
- **`optimizer::normalize_expr` is 95 lines of structural copy**
  (`optimizer.rs:93-187`) for two arms that do work: `normalize_cast` and
  `normalize_encode`. Both are encoding rewrites that belong at the point the
  encoding chain is derived, not in a separate fixed-point pass.
- **`trait Backend` has three impls and one of them decides anything.**
  `backend/host.rs` and `backend/metal.rs` return `TileLowering::default()`.
  The real rules are `backend/cuda.rs`'s `encode_rows_per_tile` and
  `encode_fusion`. A trait with dynamic dispatch, a registry (`for_backend`,
  `backend/mod.rs:89`) and a `name()` method used only by tests
  (`backend/tests.rs:240`) to express one `match`.
- **`backend/host.rs` decides nothing at all.** Host *execution* is
  `host_executor.rs` (1,019 lines); `backend/host.rs` is a null lowering wearing
  the same name.
- **Three geometry types with identical fields.** `ir::GatherDim{count,
  src_stride, dst_stride}` ≡ `load_plan::DimSpec{count, src_stride,
  dst_stride}`, converted by `gather_extents`. `ir::GatherPiece` ≈
  `load_plan::StridedExtent` ≈ `compile::Run`. One rectangle, four spellings.
- **`CompileError` has two variants and 285 `InvalidInput` sites.** The type
  carries no information; the string does.

### 2.3 Dead code, confirmed by grep

| Item | Evidence |
| --- | --- |
| `StorageInstr::Release` | Constructed **once**, in `ffi/tests.rs:232`. Handled in 8 places. |
| `TileMapKind::Reorder` | Never emitted. Exists only in `passes.rs:569` validation and `backend/cuda.rs:20`'s mask. |
| `QuantSpec::{zero_point_dtype, block_shape, scale_dtype}` | **Read zero times.** Written by 8 construction sites, the C ABI mirror and the C++ builder. |
| `Gather::zero_fill` | `contract/compile.rs` sets it, `typecheck.rs:58,86` admits it, `planner.rs:305-311` rejects it unconditionally. `Expr::Pad` is therefore **unreachable**. |
| `rewrite.rs::old_to_new` | Written at 137/152/158/174/203, never read. |
| `Backend::name()` | `backend/tests.rs:240-242` only. |

### 2.4 Optimization: compilation is superlinear, and it is two lines

Measured with a synthetic probe: *N* tensors, *N* passthrough contracts,
`[64,64]` BF16, `--release`. The probe was deleted; the two patches were
reverted; the tree is clean.

| N | today | + `tensor_by_name` index | + buffer index |
| ---: | ---: | ---: | ---: |
| 2,000 | 30.9 ms | 14.3 | 18.1 |
| 8,000 | 151.8 ms | 68.5 | 50.2 |
| 16,000 | 409.8 ms | 178.8 | 128.2 |
| 32,000 | **2,149.3 ms** | 518.3 | **329.3** |

4× the tensors costs 14× the time — about O(N^1.9). Two causes, both linear
scans inside loops:

- `checkpoint/mod.rs:40-42` — `tensor_by_name` is
  `self.tensors.iter().find(...)`, called **twice per contract tensor**, from
  `frontend.rs:359` and from the `CheckpointTypes` impl at `mod.rs:45-53`.
- `planner/passes.rs:52-58` and `passes.rs:449-458` —
  `program.buffers.iter().find()` inside the instruction loop.

**6.5× for two indices.** Remaining superlinearity: `planner.rs:744`
(`declare_view_buffer`), `planner.rs:776` (`promote_buffer`),
`backend/mod.rs:228-235` (`buffer_tensor`).

Other measured waste: `instrs.clone()` four times (`passes.rs:11, 76, 181,
601`); `rewrite.rs:27` deep-clones the entire contract tree on the `tp_size <= 1`
no-op path; `typecheck` runs three times.

### 2.5 Structural smells

- **Dependency inversion.** `load_plan.rs:9-15` re-exports
  `crate::ffi::types::PIE_LOADER_TILE_MAP_*`. The core imports its own ABI.
  `contract_writer.rs:18,24` does the same.
- **Test-only code shipped in the library.** `contract_writer.rs` (246 lines)
  and `reference.rs` (204 lines) are `pub mod` in `lib.rs` and used only by
  `ffi/tests.rs`, `tests/golden_plans.rs` and `tests/algebra.rs`.
- **`use super::*` in every planner submodule** (`arena`, `memory`, `passes`,
  `extents`) — the split is textual, not a boundary.
- **Name-string guessing, twice.** `load_plan.rs:194` `derive_quant_attachments`
  matches `_scale_inv` / `.scale` / `_scale` suffixes; `planner.rs:659`
  `block_scale_source` matches `_scale_inv`. The comment at
  `load_plan.rs:180-186` admits it is restating `frontend.rs::quant_metadata_outputs`.
  The frontend knows the relationship structurally and throws it away.
- **Untunable constants.** `passes.rs:170-180` `SlabConfig` (7 fields),
  `rewrite.rs:30-32` `MIN_GROUP_TENSORS = 16`, `DEFAULT_MAX_BANK_BYTES = 4 GiB`.

---

## 3. Seven principles

### P1 — One geometry type

`compile::Run`, `compile::Piece`, `ir::GatherPiece` + `ir::GatherDim`,
`load_plan::StridedExtent` + `load_plan::DimSpec` collapse into one `Extent`
that survives from the solver to the ABI without re-encoding.

This is also what makes spec.md §3.3's third fold rule — *the destination must
stay dense* — checkable in one place instead of re-verified at each re-encoding.

### P2 — Delete the middle algebra

`ir.rs`, `optimizer.rs` and `typecheck.rs` go away (854 lines). `contract::Expr`
is the high IR; `Lowering` is its solution; `plan/build.rs` goes straight from
`Lowering` to `StorageInstr`.

The encoding chain that `LayoutExpr::{Cast, Decode, Encode, Transcode}`
represented is **derived, not stored**: it is a function of `(source encoding,
declared encoding)`, computed at the point of use. `normalize_cast` and
`normalize_encode` become part of that derivation, which is where they were
trying to be.

### P3 — Passes are a registered list over a shared index

```rust
trait Pass {
    fn name(&self) -> &'static str;
    fn run(&self, p: &mut Program, ix: &PlanIndex) -> Result<Stats, Error>;
}
```

`PlanIndex` is built once (name → tensor, buffer → tensor, tensor → instrs) and
handed to every pass, which kills §2.4's scans by construction rather than by
patch. `planner.rs:135-146`'s hardcoded sequence becomes a `Vec<Box<dyn Pass>>`,
and pass statistics stop being ad-hoc fields on `OptimizerReport`.

Adding a pass becomes one registration line.

### P4 — The ABI is generated, and it points inward

An `abi_enum!` macro emits the `#[repr(C)]` mirror, the `From`/`TryFrom` pair
and the constant block from one declaration, replacing 19 hand-written
conversions in `ffi/types.rs` (815 lines). `flatten_instr`, `read_expr` and
`plan_view` move into `abi/{encode,decode}.rs`.

The dependency arrow reverses: `abi/` imports the core; the core never imports
`abi/`. §2.5's inversion is a compile error afterwards.

### P5 — Errors name a place and a reason

```rust
enum Error { Contract{..}, Shard{..}, Overflow{..}, Unsupported{..}, Checkpoint{..} }
```

285 `InvalidInput(format!(...))` sites become typed. The C ABI gets a real error
code instead of a string it cannot match on.

### P6 — `trait Backend` is demoted to a function

```rust
fn lower_tile(facts: &TileMapFacts, target: &StorageTarget) -> TileLowering
```

with a `match target.backend` inside. The registry, the dynamic dispatch, the
two null impls and `name()` all go.

> **Trade-off, stated explicitly.** This is the one principle that *reduces* a
> designed extension point. It is right today — three impls, one of which
> decides anything — and it is wrong the moment a fourth backend has genuinely
> different tile rules. The `match` is a two-minute reversal back to a trait if
> that day comes; the trait is not free until then. **Confirm before executing.**

### P7 — Test scaffolding leaves the library

`contract_writer.rs` and `reference.rs` move to a `testkit` target that
`tests/` and `ffi/tests.rs` depend on and `pie-loader` does not ship.

---

## 4. The algebra under v2

The operator set is the paper's contribution, so this section is about what
changes in **spec.md §3** — the answer is: almost nothing, and that is the
point. What changes is that it becomes the *only* algebra in the loader.

### 4.1 What is actually used

Counted over all 18 golden contracts (real model families) and all `contract.hpp`
author call sites:

| op | golden nodes | models | C++ sites | v2 |
| --- | ---: | ---: | ---: | --- |
| `Src` | 264 | 18 | 14 | keep |
| `Shard` | 75 | 7 | 3 | keep — the core idea |
| `Cat` | 20 | 5 | 7 | keep |
| `Reshape` | 16 | 2 | 3 | keep |
| `Repack` | 16 | 2 | 1 | keep — the one escape hatch |
| `Slice` | 6 | 1 | 3 | keep |
| `Out` | 4 | 1 | 2 | keep |
| `Bitcast` | 0 | 0 | 1 | keep — MLX path |
| **`Pad`** | **0** | **0** | **0** | **fix** (§4.3) |
| **`Quantize`** | **0** | **0** | **0** | **remove** (§4.2) |

### 4.2 `Quantize` moves from the algebra to the type

No author ever wrote it. Runtime quantization is declared by setting the
encoding on the record (`driver/cuda/src/model/contract.hpp:330`):

```cpp
return pie_loader::quantized(contract.with_block_shape(quant, {32}));
```

`frontend.rs` then derives the encoding chain from the `(source, declared)`
pair. So the honest statement is stronger than the one spec.md currently makes:

> **A change of encoding is not an operator. It is the difference between two
> types.** The algebra says only *where the bytes are*; the record says *what
> they mean*.

`Repack` remains, because it is genuinely a rearrangement the algebra cannot
denote — which is the correct criterion for an escape hatch, and one `Quantize`
never met.

Result: **nine constructors and one escape hatch**, and spec.md §2's coverage
table gains a symmetry it should always have had:

> The fourteen cases are **9 expression constructors + 3 record fields (`name`,
> `shape`, `encoding`) + 1 escape hatch.**

Case 13 (quantize) joins case 2 (rename), which spec.md already describes as
*"not an expression — the name is a record field."*

### 4.3 `Pad` must be made real

It is spec.md §2 case 9 and it has a row in the §3.3 cost table — *head-dim pad
of `[4,4]` by one column → 5* — but `planner.rs:305-311` rejects `zero_fill`
unconditionally. The compiler prices something it refuses to build.

Fix rather than delete, for three reasons: the demand is real
(`driver/cuda/src/model/llama_like/llama_like.cpp:640` — *"Pad Q/K/V to `dk`
when the model's head_dim isn't a flashinfer..."*); the cost model already
treats it correctly (*a hole is a fill, not a copy*); and §3.3's third fold rule
is an argument **about** padding, so removing `Pad` guts it.

Cost: one plan instruction, `Fill { buffer, range, value: 0 }`. This is the only
new `StorageInstr` v2 requires for correctness.

### 4.4 `Bitcast` and `Reshape` both stay — write down why

Denotationally `Bitcast ⊇ Reshape`: `Bitcast(e, T{shape, same_encoding})` ≡
`Reshape(e, shape)`. They are still two operators, because `Bitcast` must
restrict its operand to a whole tensor — when element width changes, element
offsets into a partial view denote nothing — while `Reshape` is a byte identity
and composes freely. Merging either over-restricts `Reshape` or under-restricts
`Bitcast`. spec.md does not currently say this and should.

### 4.5 The cost model must decide something

This is the largest gap between the paper and the code.

```
Lowering::cost()              → callers in src/: 0   (tests only)
Lowering::run_count()         → callers in src/: 0   (tests only)
Lowering::mean_run_elements() → callers in src/: 0   (tests only)
```

Production reads exactly two things off a `Lowering`: `pieces()` and
`needs_zero_fill()` (`frontend.rs:176`). spec.md §3.3 admits it:

> *"The threshold is a loader policy informed by measured bandwidth. **Today
> this decision does not exist**; passes hard-code one lowering each."*

**Landed differently, and the plan above was wrong.** The sketch was:

```rust
match lowering.cost() {
    c if c <= target.gather_threshold => emit_extent_writes(lowering.pieces()),
    _                                 => emit_gather(lowering.runs()),
}
```

Two things were wrong with it.

First, `Gather` already exists. It is `StorageInstr::SlabScatter` — one
over-read of a file region plus a device-side descriptor buffer of placements —
and `driver/cuda/src/loader/load_plan_executor.hpp` has had the kernel for it
all along. Adding a second instruction for the same idea would have been the
thing §4.1 is against.

Second, and this is the real error: **that decision cannot be a function of a
`Lowering`.** A slab groups writes from *different tensors*, sorted by file
offset, and what decides it is the gap between two source ranges and the ratio
of span to payload. A `Lowering` is one tensor's expression; it has neither the
gaps nor the neighbours. The quantities do not even exist until arena offsets
are assigned. So the decision belongs to a pass over the whole schedule, after
`assign-persistent-offsets`, which is exactly where
`plan/passes/rewrite.rs::build_slab_scatter_writes` already makes it. The
existing placement was right and the plan was wrong about it.

What was actually missing was the *other* direction. `cost()` had no reader,
but the decision it can make was already being made — spelled as a slice
pattern where nobody would recognise it:

```rust
if let [rect] = rects.as_slice() && rect.dst_offset == 0 && rect.bytes() == output_bytes
```

That is `cost() == 1`: one rectangle covering the whole destination, so the
tensor can alias the checkpoint's bytes or view a buffer instead of copying.
It is the most valuable choice the compiler makes — a whole-tensor load moves
nothing — and §3.3's cost model selects it. `plan/build.rs` now says so.

So §3.3's claim splits in two, and both halves are now true: the cost model
*does* decide a lowering (at cost 1), and the slab thresholds *are* a loader
policy rather than a function of the algebra — which is what §3.3's own
sentence said before the sketch above talked us out of it. They stay off
`StorageTarget`: they are not facts about a device, since every backend reads
files the same way, and §13 already has enough `TargetSpec` accumulation.

`run_count()` and `mean_run_elements()` keep no production reader and are kept
deliberately: they are measurements of the fold, they are what the §3.3 table
is built from, and `contract/compile.rs`'s tests are their readers.

### 4.6 The rewrite laws get a home

spec.md §3.4 states three identities and nothing checks any of them. With
`Quantize` moved to the type (§4.2), the third one restates as a property of the
encoding-chain derivation rather than of an operator:

| spec.md §3.4 | v2 |
| --- | --- |
| `Slice_a ∘ Cat_a` | property of `contract/affine.rs`, proved by `tests/algebra.rs` |
| `Cat_a ∘ Slice_a` | same |
| `Quantize_a ∘ Cat_a ≡ Cat_a ∘ Quantize_a` | **a declared encoding on a `Cat` distributes over its parts** — enables quantizing during a fused q/k/v read instead of staging BF16 |

The third is the missed optimization §3.4 already identifies. Under v2 it is
expressible; whether to implement it is §9.

### 4.7 Why the retired variants stay retired

`architecture.md:1459` records eight `LayoutExpr` variants retired during step
7b: `ByteSpans`, `Select`, `Partition`, `Join`, `Stack`, `Unzip`, `Reorder`,
`View`. Since v2 makes that retirement permanent by deleting the enum entirely,
the reasons are worth writing down once. They are not the same reason, and the
buckets have different futures. Definitions below are from
`git show c5fb079ab^:runtime/load-planner/src/ir.rs`.

**A — subsumed by a more general operator.** Expressive power went *up*.

| retired | replacement | what changed |
| --- | --- | --- |
| `Select{input, axis, start, length}` | `Slice(e, axis, start, len, **step**)` | `step` subsumes `RowMap::{Even,Odd}` (spec.md §2 case 8) |
| `Join{inputs, axis}` | `Cat(axis, parts)` | rename |
| `Partition{input, axis, parts, **index**}` | `Shard(e, axis)` | **the `index` field is gone** |

`Partition` is the load-bearing one. Carrying `index: u32` on the node meant the
IR knew its own rank, which meant a different contract per rank. `Shard` carries
no rank; `specialize` supplies it later. The rank-independence property that
`tests/standalone.rs` pins is precisely the removal of that one field.

**B — derivable by composition.** Expressive power unchanged, node count down.

| retired | derivation |
| --- | --- |
| `Stack{inputs, axis}` | `Cat ∘ Reshape` — a length-1 axis, then concat |
| `Unzip{input, axis, outputs}` | N independent `Slice` |
| `ByteSpans{spans, decl}` | — see below |

`Stack` is visible in the goldens; spec.md §2 case 6 (expert stack) compiles to
`Cat(0, [Reshape(Cat(0, [gate, up]), [1, 512, 128]), …])`
(`tests/golden/contracts/qwen3_moe_host.json`).

`Unzip` was the only **multi-output** affine node. It alone forced the IR to be
a multi-output DAG and every pass to handle that case. Split into N `Slice`s,
every affine node is single-output.

`ByteSpans` is a different animal: it was **the solver's output language
smuggled into the input language** — an escape hatch that let the frontend emit
raw byte ranges whenever the algebra could not express something. The same
information now exists as `Run` / `Extent` on the *other* side of the compiler.
Its existence is the original form of the edit-surface problem in §2.1.

**C — a category error, not a missing operator.**

- `Reorder{input, perm}` — transpose. Affine on indices, but it shatters run
  contiguity on flat offsets: one run becomes as many runs as the product of the
  outer dims. Keeping it would break §3.1's compilation property and §3.3's
  "pieces = cost" identity. spec.md §3.5 sends it to `Repack`, and
  `RepackLayout::{DenseRowGather, MarlinMxfp4*}` is where it lives today.
- `View{input, layout, axis, start, length}` — not a transformation at all.
  spec.md §1: *"Ownership is a loader decision… The driver never says 'make this
  a view'."* The planner now **derives** views (bank-and-view, spec.md §4.3).

**Do any come back?**

| bucket | returns? | why |
| --- | --- | --- |
| A | no | restoring them *loses* expressiveness; `Partition` would forfeit rank-independence |
| B | no | derivable; `ByteSpans` specifically would reopen the bypass around the algebra |
| C `View` | no | a layer violation, not an absent operator |
| C **`Reorder`** | **the one to watch** | today `Repack` covers it |

`Reorder` is the only real candidate, and v2 makes re-adding it *cheaper*, not
harder. Once `cost()` selects the lowering (§4.5), a general permutation has an
obvious home: the *"many pieces, irregular strides → descriptor buffer + gather
kernel"* branch is exactly what a transpose needs. The reason it is an escape
hatch today is that this branch does not exist.

So the deletions are safe for two reasons, and "we do not need it yet" is
neither of them: buckets A and B lose nothing, and bucket C costs **two files
instead of eight** to reverse (§6). The shrunk edit surface *is* the insurance
policy for the shrunk operator set.

### 4.8 Net effect on the paper

| | today | v2 |
| --- | --- | --- |
| algebras in the loader | **2** (`Expr` + `LayoutExpr`) | **1** |
| constructors | 8 + 2 escape hatches | **9 + 1 escape hatch** |
| leaves the algebra | — | `Quantize` → record field |
| unreachable | `Pad` | — (`Fill` added) |
| new plan instructions | — | `Fill` (`Gather` was already `SlabScatter` — §4.5) |
| §3.3 cost model | computed, discarded | **decides the no-copy lowering at cost 1** |
| §3.4 laws | stated | checked; one implementable |
| code behind the algebra | 8 files per primitive | **2** |

The operator set barely moves. What moves is that it becomes the only one, and
that the claim *"the algebra is small"* becomes true of the code as well as the
grammar.

---

## 5. Target module tree

```
loader/src/
├── lib.rs
├── error.rs             Error{Contract,Shard,Overflow,Unsupported,Checkpoint}   (P5)
├── extent.rs            the one geometry type                                   (P1)
├── contract/
│   ├── mod.rs           Expr, TensorContract, ModelContract, Resolver
│   ├── check.rs         the only type checker: infer + specialize + quant rules
│   ├── affine.rs        the solver — today's contract/compile.rs, barely touched
│   └── fold.rs          run folding + cost                                      (§4.5)
├── plan/
│   ├── mod.rs           StorageInstr, LoadPlan, StorageTarget
│   ├── build.rs         Lowering → instrs; reads cost(); emits Fill / Gather
│   ├── index.rs         PlanIndex, built once                                   (P3)
│   ├── pass.rs          trait Pass + registry                                   (P3)
│   └── passes/          arena · slab · extents · memory · tile
├── checkpoint/          safetensors · gguf · headers · read.rs (the one opener)
├── abi/
│   ├── mod.rs           abi_enum! — mirrors are generated                       (P4)
│   ├── encode.rs        plan → POD arena
│   ├── decode.rs        POD → contract
│   └── entry.rs         extern "C", catch_unwind
├── verify.rs
├── dump.rs
├── host_executor.rs
└── main.rs
loader/testkit/          contract_writer · reference                             (P7)
```

Gone: `ir.rs`, `optimizer.rs`, `typecheck.rs`, `backend/{mod,host,metal,cuda}.rs`
(→ `plan/passes/tile.rs`), `contract_writer.rs` and `reference.rs` (→ `testkit/`).

---

## 6. Success criteria

Measurable, checked at the end. None of these is a style judgment.

| | today | target |
| --- | ---: | ---: |
| files to add an affine primitive | 8 | **2** |
| files to add a pass | 1 + hardcoded order | **1 line** |
| files to add a plan field | 5 | **1** |
| IRs | 4 (+2 POD mirrors) | **2** (+1 generated mirror) |
| compile 32,000 tensors | 2,149 ms | **≤ 350 ms** |
| `src/` lines (excl. in-file tests) | 15,006 | **≈ 11,000** |
| `CompileError` variants / `InvalidInput` sites | 2 / 285 | **5 / 0** |
| `Lowering::cost()` production callers | 0 | **≥ 1** |
| unreachable algebra constructors | 1 (`Pad`) | **0** |
| golden plans | 14 | 14, byte-identical except step 4 |
| golden contracts | 18 | 18, byte-identical |

---

## 7. What this plan does not do

- **It does not change the boundary.** `plan = compile(source_facts, program,
  target)` and "no input is a model name" are settled
  (`architecture.md` §12 row 12). `tests/standalone.rs`'s four properties must
  keep passing unmodified.
- **It does not touch the solver.** `contract/compile.rs` is the best-designed
  file in the crate. It moves to `contract/affine.rs` and its geometry types are
  unified (P1); its algorithm is not rewritten.
- **It does not add configurability for its own sake.** `SlabConfig` and
  `rewrite.rs`'s constants become fields on `StorageTarget` because the plan
  already carries a target — not because a config file is wanted.
- **It does not change the C++ side**, except for the two new instruction
  opcodes in §4.3 and §4.5 and whatever `abi_enum!` regenerates identically.
- **It does not defend `Release`, `Reorder`, or the three unread `QuantSpec`
  fields.** They are deleted, and the ABI version is bumped once for all of it.

---

## 8. Migration order

Each step is independently landable and independently revertible. The golden
suite is the safety net: **14 plans and 18 contracts must stay byte-identical
through every step except 4**, which renumbers instruction ids and therefore
regenerates plans (contracts still must not move).

| # | Step | Golden |
| --- | --- | --- |
| 1 | `error.rs` — P5. Mechanical, large diff, no behavior change. | identical |
| 2 | `extent.rs` — P1. One geometry type end to end. | identical |
| 3 | `plan/index.rs` + `plan/pass.rs` — P3. Fixes §2.4 by construction. | identical |
| 4 | **Delete the middle algebra** — P2. `ir.rs`, `optimizer.rs`, `typecheck.rs` go; `plan/build.rs` consumes `Lowering` directly. | **regenerate plans**, contracts identical |
| 5 | `abi/` + `abi_enum!` — P4, and the arrow reverses. | identical |
| 6 | `testkit/` + dead code + `Backend` → function — P6, P7. | identical |
| 7 | `Fill` (§4.3) and `Gather` (§4.5) — the two new instructions, the first behavior change. | new goldens |
| 8 | spec.md amendments — §3 operator set, §3.3 threshold, §3.4 laws, §4.4's paragraph. | — |

Step 4 is the risky one and it is deliberately placed after 1–3, so that it
lands against typed errors, one geometry and a pass registry rather than at the
same time as them.

Steps 1–6 are pure refactor: if the goldens move, the step is wrong. Step 7 is
the only one that changes what the loader can do.

### What landed, against the table

Steps 1–6 landed as one commit rather than six, because the middle IR could not
be deleted incrementally: `frontend.rs` and `planner.rs` had to fuse in the same
edit that removed the thing they passed between them.

| # | outcome |
| --- | --- |
| 1–4 | `b36f373c6`. Goldens moved by **exactly** `optimizer` → `passes`; every instruction, buffer, offset and byte count byte-identical. The `live-normalization` pass turned out to be dead in all 14 plans. |
| 5 | partly. The arrow reversed — `plan` owns the tile-map bits and `ffi/types.rs` restates them under the C names, pinned by a `const` assertion, because cbindgen emits literals and cannot follow a path. No `abi/` directory and no `abi_enum!`: the six `ffi/` modules were judged sound in the row-13 review and a macro would have bought nothing. |
| 6 | `978257c21`. `Backend` → `match`; `Release`, `Reorder`, three `QuantSpec` fields and `old_to_new` deleted; `testkit` is a default-on feature that `worker/Cargo.toml` turns off, which makes "nothing in the driver path reaches the oracle" a build check. |
| 7 | `Fill` landed (`d3100d497`). `Gather` did **not**, and should not — see §4.5: it already exists as `SlabScatter`, and the decision the plan wanted to wire to `cost()` structurally cannot be one. Goldens unchanged, because no contract pads yet. |
| 8 | spec.md §3.3 rewritten around the split in §4.5; the `[4, 4]` pad row is now pinned by a test so the document and the compiler cannot drift. |

Two things the table did not predict.

**The measured win was not in the plan at all.** Compiling was O(N²) — 2149 ms
for 32k tensors — and the cause was six `iter().find()` calls in per-instruction
loops, every one of them looking up a dense id. `plan/index.rs` makes density an
invariant rather than a cache, and the same checkpoint now takes 138 ms.
`tests/scale_probe.rs` asserts the shape of the curve, not a wall-clock number.

**Writing the `Fill` test found a bug in a pass that already existed.**
`hoist-bulk-arena-writes` partitions the schedule into allocations, bulk writes
and everything else — so a fill landed in "everything else" and ran *after* the
writes it was meant to precede. A fill in the wrong place is the one reordering
error that is silent: the plan still validates and still has the right
instruction count. `validate-fill-order` makes it an invariant, so the next pass
that reorders is told rather than trusted.

---

## 9. Open questions

1. **P6 — demote `trait Backend`?** Stated as a trade-off in §3, not a
   conclusion. Needs a decision before step 6.
2. **The gather threshold (§4.5).** `target.gather_threshold` needs a default
   backed by measured bandwidth, not a guess. Until then, set it to infinity so
   step 7 is behavior-preserving and the `Gather` path is exercised only by
   tests.
3. **Should `Quantize ∘ Cat` (§4.6) be implemented or only checked?** It is a
   real optimization for fused q/k/v under runtime fp8, and it is the only one
   of the three laws that pays. The paper's claim is weaker if it stays a
   proposition.
4. **Does `Bitcast` need a golden?** It has one C++ call site (MLX) and zero
   golden coverage. Either add one or record that the path is Metal-only and
   covered elsewhere.
5. **Structural quant attachment (§2.5).** `frontend.rs::quant_metadata_outputs`
   knows the scale/weight relationship; `derive_quant_attachments` re-guesses it
   from name suffixes. Threading it through is a small change with a contract
   consequence — it may want a `spec.md` field.
