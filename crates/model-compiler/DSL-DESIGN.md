# The DSL decides; the driver binds

A design note for the two axes the vocabulary is missing — WEIGHT
REPRESENTATION (quantization) and TENSOR PARALLELISM — and for finishing
the rule they are instances of.

## The rule

**Nothing in the driver may choose a kernel.** A statement names the
symbol it runs; the driver resolves weight names to pointers, resolves
value ids to addresses, and calls. That is the whole of its job.

This is not new. `RUNG 5` already applied it to five kinds — `Attention`,
`CausalConv1d`, `GatedDelta`, `KvAppend`, `Swiglu` all THROW in the
class-trace walk, with the same sentence at each: *"the declaration
states the kernel"*. What is new is finishing it, and noticing that
quantization and TP are the same question rather than separate features.

Fourteen semantic kinds are still executed by the drivers. Five of them
make the driver CHOOSE:

| kind | the choice the driver makes today |
|------|-----------------------------------|
| `Matmul` | cuBLAS vs Marlin, from `layer.*_quant`; `beta` from `param0` |
| `Rmsnorm` | gemma fold vs plain, from `param0` |
| `RmsnormPerHead` | same fold, from `param1` |
| `Rope` | partial vs full, from `param1 != 0` |
| `SplitGdn` | row split vs interleaved, from the two widths |

The other nine are 1:1 and merely verbose. The five are the bug surface:
this arc found eight unstated in-place facts, an inverted `residual_add`,
and a `gate_up` the binding materialises as two buffers — every one of
them a place where the driver knew something the statement did not, and
nothing checked the two agreed.

## Quantization is a property of the WEIGHT, so the handle carries it

The DSL already does this once, and says so: `NormW` carries `variant`
and `per_head`, with the comment *"THE WEIGHT KNOWS"*. `rmsnorm(q,
&w.q_norm)` needs no variant argument because the handle has it.

So `MatW` gains a representation, and `matmul(x, &w)` is polymorphic
over it — resolving AT TRACE TIME to a stated symbol:

```
  MatW { name, width, layer, repr }

  repr = Bf16                     -> Launch("gemm::act_x_wt_bf16",   [w])
       | MarlinW4A16 { group }    -> Launch("gemm::marlin_w4a16",    [w.qweight, w.scales, w.zeros])
       | Fp8E4M3   { .. }         -> Launch("gemm::act_x_wt_fp8",    [w, w.scales])
```

Three consequences, all of them the point:

1. **The descriptor stops crossing.** No `param` says "this is int4";
   the SYMBOL says it. `make_weight_view(&wb.require(name),
   layer.q_proj_quant)` — the driver's dispatch — has nothing left to do
   and goes away.
2. **The extra tensors become operands.** A quantized GEMM needs scales
   and zero-points; those are WEIGHTS, and a `Launch` already carries a
   list of weight names (`qkv_decode_qk_norm_rope_write_kv` states two).
   Today they are reached through a per-layer struct the statement never
   mentions.
3. **The contract gets written.** `kernels::check_plan` refuses a symbol
   no `kernel!` row declares, and a row now carries an `operands![...]`
   list. Adding a scheme therefore forces someone to write down what the
   kernel takes — which is the enforcement this arc kept wishing for.

The cost is symbol count: one row per (kernel x scheme). That is the
right trade. A model spec that spells its arithmetic exactly is allowed
to be long; a driver that guesses is not allowed to be short.

## Tensor parallelism needs STATEMENTS, not a flag

Quantization changes which kernel runs. TP changes what the dataflow IS:
projections are sharded, and the shards are recombined by a COLLECTIVE
that is real device work nobody currently declares.

So TP is not a `repr` on a handle. It is:

* **sharded shapes**, resolved at trace time from facts the deployment
  already has (`tp_size`, `rank`) — the same way `gate_up_fused` and
  `kv_native_bf16` resolve. A rank's trace states ITS widths;
* **collective statements**, new DSL entries with their own symbols:

```
  all_reduce(x)                -> Launch("dist::all_reduce_bf16",     [])
  all_gather(x, axis)          -> Launch("dist::all_gather_bf16",     [])
  reduce_scatter(x, axis)      -> Launch("dist::reduce_scatter_bf16", [])
```

A collective is a `Launch` like any other: it has operands, it has a
result, and it needs a `kernel!` row stating its contract. Two things
that row must say and no existing row does — it is FIRE-WIDE (`whole =
true`, and for a stronger reason than XQA's: every rank must enter the
same collective the same number of times, so a row window that split one
rank's launch and not another's would DEADLOCK rather than compute the
wrong answer), and it is a synchronisation point, which the
graph-capture rules need to know.

### The fused landing is a GUARD, not a driver test

`llama_like.cpp` at `T > 1` already runs a fused collective —
`all_reduce_residual_rmsnorm_bf16`, which sums the shards, adds the
residual and norms in one launch — and picks it with a RUNTIME
predicate:

```cpp
  if (fused_ar && fused_ar->can_fuse_residual_rmsnorm(N, H, stream)) ...
```

Three terms, and they separate cleanly. The buffer registration and
`hidden` are load-time facts, so they resolve into the trace like every
other fact. What is left is `N` — the fire's token count — and that is
exactly `GuardPred::TokensLE`, already in the closed predicate
vocabulary and already used this way by qwen3.5's recurrence for its
three spellings.

So the text states the fused arm under the predicate and the two-step
form as the else, and the driver tests nothing:

```
  regions(t, layer, Some(out_shape),
    |ctx| ctx.arm(Fire(TokensLE(fuse_max)), || {
              all_reduce_residual_rmsnorm(&partial, &y, &w.mlp_norm, hidden)
          }),
    || { let summed = all_reduce(&partial, hidden);
         residual_add_rmsnorm(&summed, &y, &w.mlp_norm, hidden) })
```

The fused statement has TWO results, because the kernel has two effects:
the residual stream is updated in place (operand 1, which the row
aliases output 0 over) and the normed activation is written fresh. That
is the same two-result shape `norm_residual_scale_norm` already has.

`llama_like`'s `DeclineReason::NoPlan` names TP first. It closes when
these three entries exist.

## What the driver becomes

`execute_op` reduces to two arms: `Launch` (a symbol table) and the
structural kinds that are genuinely the walk's — `Guard`, `Peel`,
`HookSite`. That is D1's "one symbol-keyed driver", reached by removing
choices rather than by merging switches.

The shared-arm work already done (`SplitQkv`, `Embed`, `ResidualAdd`,
`Rmsnorm`, the epilogue gather, all on one `ArmCtx`) is the same
destination approached from the other side: those arms are already
family-blind, and they become table entries rather than cases.

## Order

1. `MatW::repr` and the quantized GEMM symbols — closes `NoPlan`'s
   quantization term, and llama_like carries five projection descriptors
   today so it is checkable.
2. The four remaining CHOICE kinds (`Rmsnorm`, `RmsnormPerHead`, `Rope`,
   `SplitGdn`) become stated symbols. Mechanical once (1) sets the
   pattern.
3. The nine 1:1 kinds follow; the driver's switch is a table.
4. Collectives + sharded shapes -> TP.
5. Vision and audio towers -> the multimodal decline.

Ground truth for behaviour is `origin:dev`, so a family's hand-written
pass has to survive only until basic behaviour and performance are
confirmed against it — not until every path it serves has a local A/B.
That is what makes deletion reachable at all; `COVERAGE.md` records what
each local gate does and does not see, which is a different question and
still worth knowing.
