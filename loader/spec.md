# Model Contract Specification

> **Status: implemented in Rust; the C++ half is not.** This specifies the
> declaration format a driver hands to the loader. The algebra, its type
> checker, its lowering, and all six of `loader/src/arch/`'s passes now run on
> it; `RuntimeTensorSource` and the hand-computed byte spans it carried are
> deleted. What has not moved is *authorship*: `arch/` still infers the
> declaration from `model_type` rather than being handed one. See §9.
>
> This is the spec for step 7b of `architecture.md` §12. It is expected to
> change as it is implemented; amend it in place rather than writing errata.
>
> Citations of the form `path:line` refer to the current tree and exist to
> ground each claim in evidence.

---

## 1. Design principle

There is exactly one question that decides where this boundary goes:

> **Put the contract at the point where the two sides' knowledge becomes
> disjoint.**

|            | Knows                                                        | Therefore decides            |
| ---------- | ------------------------------------------------------------ | ---------------------------- |
| **driver** | kernel shapes, quantization support, TP rank, perf heuristics | *what* it wants              |
| **loader** | file layout, storage bandwidth, memory budget, DMA engine     | *how* the bytes get there    |
| **either** | —                                                             | **neither needs the model family** |

The last row is the load-bearing one. Today `loader/src/abi/arch.rs` carries a
13-field `ArchProfile` keyed by `model_type`, and every field exists because one
side had to guess something the other side knew. If the driver declares what it
wants, there is nothing left to infer, and the table deletes.

Two corollaries that fall out of the principle and are worth stating up front,
because they kill concepts that look load-bearing:

- **Fusion is a driver decision, not a runtime option.** The driver knows the
  quantization scheme, the TP degree, the backend (it *is* the backend) and the
  performance characteristics of its own kernels. All four inputs to "should
  q/k/v be one buffer" are on the driver side. So it decides and declares one
  shape. There is no need for the contract to express "either of these two
  layouts is acceptable", and therefore no need for a group/alternative
  concept.
- **Ownership is a loader decision.** A contract entry says "I need a tensor
  handle named X with shape S". Whether X owns its bytes or aliases into a
  larger buffer is a memory/bandwidth trade-off, which is the loader's domain.
  The driver never says "make this a view".

---

## 2. What the contract must express

Every transformation the current tree performs, enumerated. The design is only
credible if it covers all of them.

| #  | Case                                          | Where it lives today                              |
| -- | --------------------------------------------- | ------------------------------------------------- |
| 1  | Verbatim copy                                 | `RuntimeTensorSource::DirectTensor`               |
| 2  | Rename                                        | `abi/metal.rs` `metal_qwen35_runtime_name`        |
| 3  | Concat (q,k,v → qkv)                          | `abi/fusion.rs`, `Join { axis }`                  |
| 4  | Split (qkv → q,k,v)                           | `abi/phi3.rs`, `ByteSpans`                        |
| 5  | Tensor-parallel shard                         | `Sharding` + `shard_axis`                         |
| 6  | Expert stack (`[I,H]`×E → `[E,2I,H]`)         | `abi/qwen_moe.rs` `add_qwen_moe_expert_stacks`    |
| 7  | Nested shard-inside-concat                    | `abi/qwen_moe.rs` `add_fused_moe_gate_up_tp_slices` |
| 8  | Strided select (even/odd rows)                | `RowMap::{Even,Odd}`                              |
| 9  | Zero pad to a kernel-friendly multiple        | `RepackSpec.valid_rows` vs `.target_rows`         |
| 10 | View into another output (bank + views)       | `RuntimeTensorSource::SelectContract`             |
| 11 | Alias (tied embedding / lm_head)              | ad-hoc                                            |
| 12 | Hardware swizzle (Marlin MXFP4)               | `RepackSpec` (11 fields)                          |
| 13 | Load-time quantization                        | `abi.rs` `push_runtime_quant`                     |
| 14 | Quant triplet (weight / scale / bias)         | `abi/gpt_oss.rs`, `abi/metal.rs`                  |

Two observations decide the shape of the algebra:

1. **Cases 1–11 are all the same case.** Every one of them is "output elements
   are a rearrangement of input elements". Only 12 (an opaque permutation) and
   13 (arithmetic) are genuinely different.
2. **Nothing in the tree transposes.** A grep for `transpose|permute` across
   `loader/src/` finds one unrelated comment. Every rearrangement is
   axis-preserving. This is what makes the byte-span compilation below
   tractable, and it is an empirical fact about checkpoint formats, not an
   assumption we are imposing.

---

## 3. The algebra

```text
Expr :=
    Src(name)                                 -- a checkpoint tensor, by its on-disk name
  | Out(name)                                 -- an earlier contract in the same declaration
  | Slice  (Expr, axis, start, len, step)
  | Cat    (axis, [Expr])
  | Reshape(Expr, shape)                      -- row-major: a byte identity
  | Pad    (Expr, axis, before, after)
  | Bitcast(Expr, out: TensorType)             -- same bytes, different element type
  ---------------------------------------------- affine fragment ends here
  | Repack  (Expr, spec, out: TensorType)     -- escape hatch 1: hardware swizzle
  | Quantize(Expr, spec)                      -- escape hatch 2: arithmetic
```

Seven constructors, plus two explicitly-labelled escape hatches.

`Src` and `Out` are separate on purpose. A single `Ref` with a resolution order
looks tidier, but contracts routinely re-publish checkpoint names — a bank plus
views under the *original* names (`abi/nemotron.rs:154-172`) — so one namespace
would make "the file's `q_proj`" versus "my `q_proj`" depend on declaration
order. Making the distinction explicit costs one leaf variant and removes the
whole class of question. Because `Out` may only name an *earlier* entry, the DAG
is acyclic by construction and the checker runs in one pass.

`Repack` carries its own output type. The layout transform is opaque to the type
checker by definition (Marlin swizzles reshape and re-encode), so it must
declare what it produces rather than have it inferred. This is the honest cost
of an escape hatch; the operand subtree is still checked.

`Bitcast` is the node the MLX port forced into existence, and it is worth
naming rather than absorbing. A checkpoint stores 4-bit MLX weights eight to a
`u32` word, so the file declares `[rows, cols/8]` of `u32` while the runtime
wants `[rows, cols]` of `Quant(MlxAffineU4)`. Nothing is moved and nothing is
computed; only the reading changes. The alternative — letting the declared shape
simply disagree with the source shape and suppressing the check — is what the
code did before, and it disabled the type checker for the one case that most
needed it.

Two restrictions keep it honest. The two types must have the same byte size,
which is what makes it a *re*-interpretation rather than a reinterpretation plus
a silent truncation. And the operand must be a whole tensor (`Src` or `Out`),
because once the element width changes, an element offset into a partial view
denotes nothing: `Slice(e, 0, 3, 1)` before the cast and after it are different
byte ranges, and no rule picks one.

Coverage of §2:

| Case                     | Expression                                         |
| ------------------------ | -------------------------------------------------- |
| 1 verbatim               | `Src(n)`                                           |
| 2 rename                 | not an expression — the name is a record field     |
| 3 concat                 | `Cat`                                              |
| 4 split                  | `Slice`                                            |
| 5 TP shard               | `Slice(e, axis, rank*len, len)`                    |
| 6 expert stack           | `Cat(0, [Reshape(e_i, [1,I,H]) for i])`            |
| 7 nested                 | `Cat` ∘ `Slice` ∘ `Cat` — plain composition        |
| 8 strided select         | `Slice(e, 0, 0, n, step=2)`                        |
| 9 pad                    | `Pad`                                              |
| 10 view of another output| `Out`                                              |
| 11 alias                 | `Out`                                              |
| 12 swizzle               | `Repack`                                           |
| 12b packed sub-byte read | `Bitcast`                                          |
| 13 quantize              | `Quantize`                                         |
| 14 quant triplet         | three contracts sharing an `Encoding::Quant` spec  |

### 3.1 The compilation property

> **Every expression in the affine fragment denotes a piecewise-affine partial
> index map from output coordinates to source byte offsets, and the number of
> pieces is a computable function of the expression.**

Each constructor contributes:

- `Src` / `Out` — the identity map.
- `Slice` — affine (one piece per iteration of a non-innermost strided axis).
- `Cat` — piecewise-affine, one piece per part.
- `Reshape` — identity on bytes; row-major dense layout makes it free.
- `Pad` — *partial*: padded coordinates have no source and read as zero.

Composition of piecewise-affine partial maps is piecewise-affine partial, so
the fragment is closed. Therefore the loader can compile any expression in it
directly into a set of `(src_offset, dst_offset, len)` spans, **without ever
materializing an intermediate buffer** — no matter how deeply nested.

This is what makes the design practical rather than merely tidy: `RuntimeByteSpan`
stops being something a driver-facing pass constructs and becomes what the
compiler *emits*.

### 3.2 The type checker earns its keep

Implemented in `loader/src/contract.rs`. Beyond propagating shapes it enforces
the invariants that keep a quantized encoding meaningful, because a
block-quantized tensor's scales only line up with its data on block boundaries:

- `Slice` along the encoding's `channel_axis` must have `step == 1` and both
  `start` and `len` a multiple of `group_size`.
- `Cat` along the `channel_axis` requires every part's contribution to be a
  multiple of `group_size`.
- `Pad` along the `channel_axis` must pad by a multiple of `group_size`.
- `Quantize` requires the grouped axis extent to be a multiple of `group_size`.
- `Reshape` of a quantized operand is rejected outright, and `Quantize` of an
  already-quantized one likewise. No production case needs either; see §8.

These are cheap, they are checked before any I/O, and each one corresponds to a
class of silently-wrong load that the present pass pipeline can produce.

### 3.3 The cost model is in the grammar

The cost is computable before any I/O happens, so the loader has a real
planning decision rather than a hard-coded strategy. But "cost" has to be
counted in the units the executor actually charges, and the obvious count is
the wrong one.

Compiling an expression gives a list of **runs**: maximal stretches of the
output that come from one contiguous stretch of one source. Runs are the
semantics, and they are what can be checked against a reference resolver
(§9.1). They are *not* the cost. A row shard — `slice` on the innermost axis of
a `[2048, 6144]` weight — is one run per row, 2048 of them. Charging 2048 for
it would be wrong by three orders of magnitude, because every row has the same
length and both the source and the destination advance by a fixed stride. The
executor issues that as **one** strided copy, and the low IR already says so:
`load_plan::StridedExtent` carries a `DimSpec { count, src_stride, dst_stride }`
per loop level.

So the runs are folded back into loop nests before anything is counted. The
fold is a repeated sweep over the run list: find maximal consecutive groups of
same-source, same-extent items whose source and destination offsets are both in
arithmetic progression, and wrap each group in one more loop level. Because it
only ever rewrites a verified run list into a form enumerating the same
`(src, dst)` pairs, it cannot change the meaning — which is asserted directly,
by expanding both forms and comparing.

Three rules make the resulting number honest:

- **A hole is a fill, not a copy.** `Pad` does not copy zeros into the
  destination; the destination is zeroed once and then not written.
- **Different sources never fold.** Fusing q, k and v costs three copies
  however the shards are shaped, because three separate regions of the
  checkpoint are being read.
- **The destination must stay dense.** A group only folds when its destination
  step equals the size of one item, so a piece always writes one contiguous
  output range. A loop that skips in the *destination* is a scatter, and no
  copy engine below this layer can execute one: both drivers reject a
  non-compact `dest.stride` outright, and the strided path builds a compact
  staging block before it copies. The source may skip freely — that is what a
  shard is.

The third rule is what keeps the cost model from lying about what will happen.
It costs something: a head-dim pad lands its surviving bands in a destination
that skips, so it is one copy per band rather than one for all of them.
Lifting it means teaching both drivers a scattering copy, which is tracked in
§8; until then the model reports what the machine will actually do.

What the model then reports:

| expression | runs | cost |
| --- | ---: | ---: |
| whole tensor | 1 | 1 |
| row shard of `[2048, 6144]` | 2048 | 1 |
| strided expert select | one per expert | 1 |
| q/k/v fusion | 3 | 3 |
| q/k/v fusion, each shard strided | 1024 | 3 |
| head-dim pad of `[4, 4]` by one column | one per band | 5 |

And the decision it feeds:

- **few pieces** → DMA / `cudaMemcpy` per piece.
- **many pieces** — an expression whose runs genuinely do not fold, because the
  strides are irregular rather than merely numerous → materialize a descriptor
  buffer and run a gather kernel.

The threshold is a loader policy informed by measured bandwidth. Today this
decision does not exist; passes hard-code one lowering each.

### 3.4 Rewrite laws

Because the algebra is small and total, useful identities are checkable:

```text
Slice_a ∘ Cat_a       ≡  Cat_a of the surviving parts, re-sliced   (drops whole parts)
Cat_a   ∘ Slice_a     ≡  Slice_a ∘ Cat_a                            (when ranges tile)
Quantize_{axis=a} ∘ Cat_a  ≡  Cat_a ∘ Quantize_{axis=a}
```

The third one is not academic. `abi.rs:592` sets `channel_axis: Some(Axis(0))`
for both `Fp8E4M3` and `Int8Symmetric` — scales are per output channel, which is
the same axis q/k/v are concatenated along. So quantization commutes with the
fusion, meaning the loader may quantize *during* the read instead of staging
BF16 first.

That in turn proves something about the current code: `abi/fusion.rs:28-33`
bails out of fusion entirely when `runtime_quant_enabled`. That was never a
correctness requirement — it is a **missed optimization**, and the present
design has no vocabulary in which to even ask the question.

### 3.5 What is deliberately *not* in the algebra

- **Transpose / permute.** Affine, but element-granular as bytes, so it would
  break the "pieces = spans" cost model. Nothing needs it (§2). If a backend
  ever does, it belongs in `Repack`.
- **Arbitrary index expressions** (Halide/TVM style). General enough to express
  anything, which means the loader could no longer plan DMA without whole-program
  analysis. The restriction to axis-preserving maps is the whole point.
- **Alternatives / "either layout is fine".** See §1: the driver has all the
  information needed to choose, so optionality would only move a decision to the
  side that knows less about it.

---

## 4. The record

```rust
pub struct TensorContract {
    /// Binding key. What the driver will look this up by.
    pub name: String,
    /// Where the bytes come from.
    pub expr: Expr,
    /// Expected logical shape. Redundant with `expr` — see below.
    pub shape: Vec<i64>,
    /// Element type and, if quantized, how to interpret it.
    pub encoding: Encoding,
}

pub struct ModelContract {
    pub abi_version: u32,
    /// Target property; no reason to repeat it per tensor.
    pub alignment: u32,
    /// A DAG in declaration order: an entry's `Out` refs may name only
    /// earlier entries, so it is acyclic by construction.
    pub tensors: Vec<TensorContract>,
}
```

Four fields per tensor, down from nine in `RuntimeTensorContract`.

### 4.1 Why `shape` is intentionally redundant

`shape` is derivable from `expr`. It is declared anyway, and the loader
**checks** it. This makes every contract entry simultaneously a *request* and a
*proof obligation*: if the driver's mental model of the checkpoint disagrees
with what the checkpoint actually contains, the load fails at compile time with
both numbers in the message, rather than producing a plausible-looking buffer
that computes garbage.

Combined with the existing coverage check, the property is:

> **If it compiles, the bind cannot be silently wrong.**

This is the single strongest reason to prefer this design over one that merely
threads byte spans across the FFI, and it costs one `Vec<i64>` per tensor.

### 4.2 Fields that disappear, and why

| Removed                        | Reason                                                        |
| ------------------------------ | ------------------------------------------------------------- |
| `dtype`                        | already in `Encoding::Raw(d)` / `Quant(spec).logical_dtype`    |
| `metadata: Vec<TensorId>`      | never populated at any construction site; dead                 |
| `layout.strides`               | always empty; `Layout::dense()` is the only constructor        |
| per-tensor `alignment`         | always `target.preferred_alignment.max(1)`; hoisted to header  |
| `sharding` + `shard_axis`      | a TP shard **is** a `Slice`                                    |
| `RuntimeTensorSource::SelectContract` | an `Out` ref — a DAG edge                               |
| `RuntimeByteSpan` in the contract | now the compiler's *output*, not a driver's input           |
| `consumed: HashSet` + pass order  | a declaration has no evaluation order                       |

### 4.3 The DAG makes bank-and-view stop being a pattern

Because `Out` names a previously-declared contract, "materialize once, publish
views into it" is not a special construction — it is two entries, one referring
to the other, and the loader's job of not duplicating bytes is ordinary
common-subexpression sharing. The driver never asks for a view; it asks for a
tensor and the loader notices it can alias.

Three places in the current tree independently invented this mechanism:

- `abi/nemotron.rs:123-172` — a packed bank plus `SelectContract` views.
- `abi.rs:392-424` `coalesce_direct_row_shards` — a row-shard bank plus views.
- `driver/cuda/src/loader/tensor_spec.hpp` — `TensorDecl` grew
  `backing_tensor` / `view_axis` / `view_start` / `view_length` on the C++ side.

A fourth existed in the deleted `semantic.rs` (`SemanticGraph { tensors, groups }`).
`abi/fusion.rs:105-119` is the outlier that pushes a `Join` and *no* views,
which is exactly why `bind_projection_or_fused_view`
(`driver/cuda/src/model/llama_like/qwen3.cpp:24-56`, 32 lines) has to
re-derive q/k/v offsets from `HfConfig` in C++.

---

## 5. Worked example — Qwen3-1.7B, layer 0, TP=2 rank=0, fp8

Measured shapes: `hidden=2048`, `n_q=16`, `n_kv=8`, `head_dim=128`, so
`q_proj[2048,2048]`, `k_proj[1024,2048]`, `v_proj[1024,2048]`.
At TP=2 rank 0 the local halves are 1024 / 512 / 512 rows, fusing to 2048.

```rust
// Builders on Expr; see loader/src/contract.rs.
let qkv = Expr::cat(0, vec![
    Expr::src("model.layers.0.self_attn.q_proj.weight").slice(0, 0, 1024),  // [2048,2048] -> [1024,2048]
    Expr::src("model.layers.0.self_attn.k_proj.weight").slice(0, 0,  512),
    Expr::src("model.layers.0.self_attn.v_proj.weight").slice(0, 0,  512),
]);

TensorContract::new(
    "model.layers.0.self_attn.qkv_proj.fused.weight",
    qkv.quantize(fp8_e4m3_per_row()),
    vec![2048, 2048],
    Encoding::Quant(fp8_e4m3_per_row()),
)
```

This combination is **impossible to express today**. `abi/fusion.rs:28-33`
abandons fusion if *any* of `tp_size != 1`, `runtime_quant_enabled`, or
`profile().skip_dense_qkv_fusion` holds. Three independent refusals, all of
which are composition in the algebra.

Phi-3 is the same declaration read in the other direction — the checkpoint ships
one fused qkv and the driver wants three tensors:

```rust
// Hq and Hk come from the driver's own HfConfig, which it already reads. The
// loader learns nothing about attention; it sees three slices of one tensor.
TensorContract::new("…q_proj.weight", Expr::src("…qkv_proj.weight").slice(0, 0,     Hq), vec![Hq, H], bf16)
TensorContract::new("…k_proj.weight", Expr::src("…qkv_proj.weight").slice(0, Hq,    Hk), vec![Hk, H], bf16)
TensorContract::new("…v_proj.weight", Expr::src("…qkv_proj.weight").slice(0, Hq+Hk, Hk), vec![Hk, H], bf16)
```

`abi/phi3.rs` and the `phi3_fused_splits` profile flag both evaporate.

---

## 6. What this deletes

| Deleted                                                       | Lines | Why                                                        |
| ------------------------------------------------------------- | ----: | ---------------------------------------------------------- |
| `loader/src/abi/arch.rs` — `ArchProfile`, 13 fields            |   174 | nothing left to infer from `model_type`                     |
| `skip_dense_qkv_fusion`                                        |     — | a C++ binder gap encoded as an architecture property        |
| `phi3_fused_splits`                                            |    94 | the same declaration in the opposite direction              |
| `bind_projection_or_fused_view` (C++)                          |    32 | views are published by the loader for free                  |
| `mla_fused_joins`, `nemotron_packed_experts`, `stack_per_expert_moe`, `gpt_oss_mxfp4_groups`, `metal_qwen35` | ~900 | declarations, not inferences |
| `Sharding`, `shard_axis`                                       |     — | a shard is a `Slice`                                        |
| `SelectContract`                                               |     — | an `Out` ref is a DAG edge                                  |
| `metadata`, `Layout.strides`, duplicated `alignment`           |     — | dead fields                                                 |

Most of `RepackSpec`'s 11 fields are absorbed by `Slice` / `Pad` / `Reshape`,
leaving only the `layout` enum.

---

## 7. Properties worth claiming

1. **A closed minimal algebra.** Five constructors cover 14 production
   transformation cases across eight model families; everything else is behind
   two explicitly labelled escape hatches. The design can state what it *cannot*
   express, which an ad-hoc pass pipeline cannot.
2. **The cost model lives in the grammar.** Span count is computed from the
   expression before any I/O, so DMA-vs-gather becomes a principled decision
   rather than a hard-coded lowering.
3. **Rewrite laws.** §3.4 — the loader becomes an optimizer over a declared
   program rather than a fixed sequence of passes.
4. **Self-verification.** `shape` is redundant on purpose, so every entry is an
   assertion; with coverage checking, a successful compile rules out silent
   misbinding.
5. **Model-family blindness.** Neither side reads `model_type`. Supporting a new
   architecture is zero lines of Rust.

---

## 8. Open questions

- **`Repack`'s layout enum is still backend vocabulary,** and it has to declare
  its own output type because the swizzle is opaque. Marlin layouts are not
  affine, so they cannot come inside the fragment. This is the last remaining
  backend coupling and the spec says so plainly rather than pretending
  otherwise. Open: whether the 11-field `RepackSpec` really reduces to
  `layout` once `Slice`/`Pad`/`Reshape` absorb the row/col arithmetic, or
  whether some parameters survive.
- **Who writes `Expr`.** The wire format is the algebra, but a driver
  hand-building trees would be miserable. A C++ builder façade —
  `decl(name).concat({q,k,v}).shard(0, rank, world).quantize(fp8)` — is required
  for this to be usable, and its design is not yet settled.
- **Policy for undeclared checkpoint tensors.** Silently ignored today (Metal
  Qwen3.5 skips the visual tower and `mtp.`). Should this become an explicit
  field so that a typo in a name is not indistinguishable from an intentional
  skip?
- **`Pad`'s fill value.** Zero everywhere today. Parameterize, or fix at zero
  until something needs otherwise?
- **`Pad` has no instruction to lower to.** The storage program has no `Fill`,
  and no `BufferDecl` flag that says "hand me this zeroed", so the frontend
  rejects any expression whose lowering needs one. Adding it touches the FFI
  POD surface and both drivers' executors. Nothing in production pads today;
  the algebra is ahead of the runtime here on purpose.
- **A copy that skips in the destination.** Both executors require a compact
  `dest.stride`, which is what forces §3.3's third folding rule. Teaching them
  a scattering write would let a padded tensor, and a concatenation on any axis
  but the outermost, collapse to one instruction instead of one per band.
- **`Reshape` of a quantized tensor is rejected for now** (§3.2), on the grounds
  that no production case needs it and that a row-major reinterpretation is not
  a byte identity once elements are packed into blocks. If a case appears, the
  rule to define is its interaction with `QuantSpec.block_shape`.
- **Where the `-1` wildcard in `Reshape` belongs.** Implemented for ergonomics,
  but it weakens the "declaration is a proof obligation" property slightly: an
  inferred extent cannot disagree with anything. Possibly it should only be
  allowed in the builder, not the wire format.

---

## 9. Implementation status

| Piece | State |
| --- | --- |
| `Expr`, `TensorType`, `TensorContract`, `ModelContract` | **done** — `loader/src/contract.rs` |
| Shape/encoding inference and the declared-type assertion | **done** — 26 tests |
| Quantization block-alignment checks (§3.2) | **done** |
| Lowering the affine fragment to runs | **done** — `loader/src/contract/compile.rs`, 26 tests |
| Folding runs into strided loop nests (§3.3) | **done** — `Lowering::{pieces, copy_pieces, needs_zero_fill}` |
| The cost model and the DMA-vs-gather choice | **done** — `Lowering::cost`; `compile`'s `max_runs` is a compile-time guard, not the cost |
| The HIGH IR node the fragment lowers to | **done** — `ir::LayoutExpr::Gather`; eight affine variants deleted with it |
| Wiring the loader's frontend, optimizer, type checker, planner and evaluator onto it | **done** — `loader/tests/algebra.rs`, 13 tests |
| Porting `arch/`'s six passes onto declarations | **done** — all six emit `Expr`; `RuntimeTensorSource`, `RuntimeByteSpan`, `SelectContract` and `frontend::bridge_expr` deleted |
| Retiring the loader's `config.json` read | **done** — `PieLoaderModelSpec` on the request; both drivers fill it |
| Checking a plan against the contract | **done** — `loader/src/verify.rs`, `verify(&PlanView, Option<&ContractView>)` |
| Golden plans | **done** — `loader/tests/golden_plans.rs`, 10 plans across 4 architectures, 2 backends, 3 TP configs |
| The loader as a tool | **done** — `loader/src/main.rs`: `dump · verify · diff · replay` |
| C++ builder façade and the FFI shape | not started — see §9.2 |

The record and the algebra are expected to keep moving while the remaining
pieces are written. Amend this document in place.

### 9.2 What the C++ façade is still for

With the passes ported and `config.json` retired, the loader no longer reads
anything about the model from disk: the driver states the five facts it needs
(`PieLoaderModelSpec`). What has *not* moved is the declaration itself. `arch/`
still decides, from `model_type`, which tensors a model wants and in what shape,
and that decision belongs to whoever knows which kernels will read them.

The remaining evidence that this is the right end state is `ArchProfile`'s four
layer-3 gates — `skip_dense_qkv_fusion`, `stack_per_expert_moe`,
`phi3_fused_splits`, `metal_qwen35`. Each exists because a pass guesses at what
the driver wants and the guess is wrong for some architecture. `fusion.rs`
invents the name `...self_attn.qkv_proj.fused.weight` and hopes something binds
it; `skip_dense_qkv_fusion` is the list of models where hoping failed. None of
these survive a world where the driver names the tensor it intends to bind.

Until then, the contract the loader authors is still a contract *someone*
authored separately from the plan, which is why `verify`'s contract check is
already worth running (§7): `arch/` and the `frontend → optimizer → planner`
pipeline are different code, and a bug in the second cannot hide in the first.
What it cannot yet catch is `arch/` and the driver disagreeing, because only one
of them currently speaks.

### 9.1 How the lowering is trusted

`compile` walks the output in flat order. At each position it resolves the
source and derives an *analytic* bound on how far that mapping stays
contiguous, emits one run, and jumps. The bounds are the delicate part, so they
are checked two ways.

*Differentially*, against a reference resolver that maps one coordinate at a
time straight off the `Expr` with no arena and no span reasoning
(`runs_agree_with_the_oracle`). Twenty expressions — every constructor, both
reshape directions, `Slice` above `Cat`, `Pad` under `Cat`, three-level
nesting — must agree element for element.

*And for maximality*: the runs must tile the output with no gap, no overlap and
no slack, and no neighbouring pair may be mergeable. A correct-but-fragmented
lowering passes every other assertion while quietly wrecking §3.3, so it is
asserted separately.

The fold on top is checked the same way, and it is the cheaper check of the
two: expand the loop nests and expand the runs, sort both, require equality.
`copy_pieces` is held to the same standard against the non-hole pairs. Every
expression in the oracle set goes through both.

`the_cost_model_charges_for_copies_not_for_runs` pins the numbers in §3.3's
table, including the one that matters most — a 2048-run row shard costing 1.

One layer down, the same expressions are checked again in *bytes*.
`loader/tests/algebra.rs` compiles each one into a real `LayoutExpr::Gather`,
runs the plan through the type checker and the optimizer, replays it with the
reference evaluator, and compares against a hand-written per-coordinate model
of the same expression. Everything above this point reasons in elements, so a
scaling mistake at the element/byte boundary would otherwise survive every
check in §9.1 — and one did, until this test found it.

`ffi::tests::a_real_checkpoint_compiles_to_a_plan_that_verifies` closes the
loop on real weights: four checkpoints at four tensor-parallel configurations,
each compiled, verified, and held to a budget of eight storage instructions per
finalized tensor. That budget is the standing guard on §3.3 — if the fold ever
stops working, the plan grows by three orders of magnitude and nothing else
would notice until a model took minutes to load.
