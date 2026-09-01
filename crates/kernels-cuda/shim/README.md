# `shim/` — headers that answer for a compiler that is not here

Every file in this directory wears somebody else's filename on purpose.

NVRTC before CUDA 13.3 ships **no device headers at all**. A `#include
<cuda_fp16.h>` under it does not find a smaller `cuda_fp16.h`; it finds
nothing, and the compile stops. The device text this crate carries — ours and
FlashInfer's alike — is written against those names, so the names have to be
answered. The files in this directory are the answer.

Fourteen of them arrived together, when this directory was made out of two
others; that is the next section. The other nine arrived one at a time and
are in **Every addition and retirement, in order** below and in **Added for
the all-reduce** under it, and five have since been retired. **This page does not restate a total**, because the
total moves and a stale one reads like a measurement: count the directory.
This page is the register for the directory: a file sitting here without an
entry is the defect, not the file.

## The name is the contract

Nothing here may be renamed. NVRTC has no `-I` and no search list: it matches
the **literal string in the `#include` directive** against the names in
`includeNames[]`. A file called `cuda_fp16.h` is answering for NVIDIA's
`cuda_fp16.h` and stops answering the moment it is called anything else.

That is also why moving them here cost nothing. The carried set in
`src/source.rs` names a file by its path relative to the tree it sits in, so
`shim/cuda_fp16.h` is carried as `cuda_fp16.h` exactly as `csrc/src/cuda_fp16.h`
was. **A compile cannot tell this move happened.** It is the one role in the
crate with that property, which is why it moved first.

## Why they are together, and what they were before

They used to be split across two directories by **who wrote them**:

| was                  | files                                                                                                     | why it was there                |
| -------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `csrc/src/` (8)      | `cuda_fp16.h` `cuda_bf16.h` `cuda_fp8.h` `cuda_fp4.h` `cooperative_groups.h` `cuda/cmath` `cuda/pipeline` `cuda/std/limits` | PIE wrote them                   |
| `csrc/vendor/` (6)   | `cstdint` `type_traits` `bit` `cuda.h` `cuda_runtime.h` `boost/math/ccmath/fabs.hpp`                       | they arrived with the vendoring |

Same job, two directories, and the only thing the split recorded was a fact
about history. A compile does not care who wrote a shim; it cares that the
name resolves. Filed by **role**, all fourteen are one thing.

Both directories in that table are historical now, not only the first. There
is no `csrc/vendor` any more, and no `csrc/` either: the FlashInfer and XQA
text was internalised, and the crate's device text is `kernels/` beside this
directory. Every
`csrc/vendor` below — the `-I` recipe in the CUTLASS section, the
`cutlass_extensions` note at the end — is quoting a tree that existed when
the measurement was taken, and is left in the words it was measured in.

The six on the second row have a second story worth keeping: every one of them
was a `#ifndef __CUDACC_RTC__` guard first, and every one of those guards was
measured and refused. Guarding `#include <cstdint>` compiles and then deletes
`uint32_t` from 2,512 device declarations. Guarding `<cuda_runtime.h>` deletes
`ushort`, which `math.cuh`'s `ex2.approx.f16` wrapper is written in. Guarding
`<cuda.h>` silently unsets `CUDA_VERSION` and the fp4 vector types disappear
with no diagnostic at all. A host header whose names reach device code is
**carried**, under the exact spelling the directive uses — the rule that is
the difference between 33 guards in the vendored tree and roughly seventy.

## Every addition and retirement, in order

Fourteen is where this directory started and twenty-one is what it had
held when the retirements began; two more arrived with the all-reduce, which
is the section after this one. The seven additions of the first wave are
below and the five retirements are the last two sections on this page. **Six of the additions were made
without an entry here**,
which is the failure this page exists to prevent: a shim with no register is
fourteen files and a habit. They are written up after the fact, from each
file's own banner and from the commit that added it, and one of them
(`cuda_runtime_api.h`) already had its section further down and keeps it.

| added | file | what it answers | the site that asked |
| --- | --- | --- | --- |
| `8f7ac3601` | `cassert` | nothing — NVRTC's preamble already supplies `assert`; only the *name* was missing | `xqa/barriers.cuh:19`. `xqa/utils.h:5` was the second asker and its include is now inside a `// PIE: REMOVED` marker |
| `c13deb22d` | `cstddef` | `std::size_t`, qualified because we are the ones who spelled it that way | `attn/attention_score_capture.cuh:27` |
| `d69beb982` | `cstring` | `std::memcpy`, as a byte loop | `attn/attention_mla_naive.cuh:63`, used at `:294-295` to pack two `__nv_bfloat16` for `mma.sync` without aliasing |
| `d69beb982` | `math_constants.h` | one `#define`: `CUDART_INF_F` | `attn/attention_mla_naive.cuh:67`, the online-softmax running maximum's identity |
| `d69beb982` | `cuda_pipeline.h` | the three `cp.async` primitives, one inline PTX instruction each | `attn/attention_mla_naive.cuh:66`, `mla_mma_paged_kernel`'s KV staging |
| `854a508df` | `cuda_runtime_api.h` | the whole file, forwarding to `cuda_runtime.h` | `cute/util/debug.hpp:38` — see **Added for CUTLASS** below, which is its entry |
| `c6c3ceab1` | `limits` | `std::numeric_limits<float>::infinity()` and nothing else | FlashInfer's MoE glue activation adaptors |

All seven landed on 2026-08-14, within one working day of each other, which is
why no single commit felt like a change to this directory. Each file's own
banner carries the measurement; this table carries only that it happened.

**Retirements** are the four sections at the end: seven `cutlass/`, `cute/`
and `cutlass_extensions/` files added and deleted the same day, `cstddef`'s
`offsetof` — the file stays, the member does not — `cuda/pipeline`, the first
file to leave this directory as a file, and the four that a deletion probe
found in one run. One of those four, `limits`, is worth a second look before
the next reader deletes something for having no asker: it went out the same
day the all-reduce came in, and the all-reduce's own closure does not want it
back.

## Added for the all-reduce, 2026-08-16

`kernels/flashinfer/comm/` was internalised — the vllm P2P all-reduce
and the trtllm fused residual+RMSNorm landing, which are the two device
kernels tensor parallelism needs — and it asked for **two new files and
seventeen new names across three existing ones**: five in `cuda_fp16.h`,
eight in `cuda_bf16.h`, four in `cooperative_groups.h`. Every asking site is
in one of those two comm headers unless the row says otherwise, and every one
of them was found the way this directory says to find them: by compiling,
reading the error, and adding the name it asked for.

| file | what was added | the site that asked |
| --- | --- | --- |
| `array` | **new file.** `std::array<T, N>` — the member array, `operator[]`, `size()` | `trtllm_allreduce_fusion.cuh`'s `allreduce_fusion_kernel_twoshot_sync`, whose PARAMETER LIST is `std::array<int, NRanks>` twice. Upstream reaches the name transitively through `<tuple>`; the vendored copy names `<array>` under a `// PIE:` marker |
| `cuda/std/optional` | **new file.** `cuda::std::optional<T>`, `nullopt`, `value_or` | `utils::get_sf_out_offset_128x4` and `cvt_quant_to_fp4_get_sf_out_offset`, the FP4 scale-factor addressing. Compiled, never reached: `comm::INSTANTIATED` holds one pattern and its quant type is `kNone` |
| `cuda_fp16.h` | `__hadd` | `vllm_custom_all_reduce.cuh`'s `assign_add(half&, half)`, the scalar step of `packed_assign_add` |
| `cuda_fp16.h` | `__half2half2`, `__habs`, `__habs2` | `maths::cuda_cast<half2, half>` and `maths::cuda_abs` |
| `cuda_fp16.h` | `__half_raw(unsigned short)`, and its `operator __half()` became `constexpr` | `neg_zero<half>`'s `static constexpr __half value = __half_raw{0x8000U}` |
| `cuda_bf16.h` | `__hadd` | `assign_add(nv_bfloat16&, nv_bfloat16)`, the same site at the other format |
| `cuda_bf16.h` | `__bfloat162bfloat162`, `__habs`, `__habs2`, `__hmax`, `__hmax2` | `maths::bf162bf162`, `cuda_abs` and both `cuda_max` overloads |
| `cuda_bf16.h` | `__bfloat16_as_ushort` | `is_negative_zero<__nv_bfloat16>` — **live**, run over every element the one-shot Lamport kernel loads |
| `cuda_bf16.h` | `__nv_bfloat16_raw(unsigned short)`, `constexpr` conversion | `neg_zero<nv_bfloat16>`, the Lamport protocol's empty-slot sentinel |
| `cooperative_groups.h` | `cluster_group`, `this_cluster()`, `grid_group::cluster_rank()`, `grid_group::num_clusters()` | `IndexHelper` and `FusedOp::rms_norm`, both inside `#if __CUDA_ARCH__ >= 900`. **Assembles at `sm_90a` and has never been run**: the one GPU here is an sm_89 L40S, which cannot launch a cluster. That file's banner says so at length |

**The one change OUTSIDE this directory belongs in this table anyway**, because
a reader chasing a missing name will land here first: `kernels/prelude/device.cuh`
gave `bf16` and `f16` `constexpr` bit constructors (which the two `_raw`
additions above need in a constant expression) and an `operator=(float)`
(which `vec_add`'s `ret[i] = float + float` needs, because `vec_t::operator[]`
returns `T&` and no constructor participates in an assignment). **`explicit`
was not relaxed on anything**: `bf16 x = 1.0f;` is still refused, which is the
property `MODIFICATIONS`' "THE EDIT THAT IS NOT A REMOVAL" turns on.

**Two things were asked for and REFUSED** -- one name and one whole class of
conversion -- and both refusals are the rule this
page states rather than an exception to it:

* `__nv_cvt_float2_to_fp4x2`, asked for by `fp32_vec_to_e2m1`.
  `cuda_fp4.h`'s banner already decided this one: a fp4 conversion is written
  *"on hardware that can be measured"*, and this box cannot run the
  instruction the software path emulates. The two overloads that call it have
  no caller and are removed with a marker naming this paragraph.
* Implicit conversions between the integer types and `f16`/`bf16`, asked for
  by six `maths::cuda_cast` specialisations. Supplying them means relaxing the
  prelude's `explicit`, which reaches every FA2 and XQA instantiation in the
  crate. The six have no caller and are removed with a marker.

Both are `MODIFICATIONS`' *"TWO REMOVALS TAKE DEVICE TEXT"*, which is the
first time this tree has removed anything from the device text that was not
host C++,
and it is written up there rather than here because the removal is in
FlashInfer's bytes and this page is about ours.

## Added for the tiled affine point, 2026-08-31

`kernels/linear/tiled.cuh` — the W4A16 post-affine tiled GEMM (§J4 hybrid,
phase A) — is the first device text in this crate to do bf16 arithmetic on a
PACKED pair for a numeric reason rather than a convenience one, and it asked
for **two names in one existing file**. Both were found the way this page says
to find them: by compiling, reading the error, and adding the name it asked
for.

| file | what was added | the site that asked |
| --- | --- | --- |
| `cuda_bf16.h` | `__hfma2` | `linear/tiled.cuh`'s `fold_post` — `w = s·c + b` folded into the B fragment. `fma.rn.bf16x2` from sm_80, and the ONE rounding is the whole point: the tiled point's golden is a host fold that computes `s·c + b` wide and rounds once, so a `__hmul2` plus an add would answer a different number and could not be held against it |
| `cuda_bf16.h` | `__hsub2` | `linear/tiled.cuh`'s `dequant_u4_bf16x2` epilogue — marlin's lop3 lands a four-bit code as `128 + code` (the `0x4300` magic exponent) and this takes the 128 back off. Exact for every input it sees: 128..143 all fit bf16's seven mantissa bits. `sub.rn.bf16x2` is sm_90, so the sm_80 body is a sign-flip and an FMA against `(1.0, 1.0)` |

**`__hneg2` and `__halves2bfloat162` were NOT added**, though the §J4 recon
predicted all four. The recon was reading marlin's `scale_and_sub`, which
negates a zero point before folding it; this point folds a post-offset bias
and ADDS it, so the negation never happens, and the pair broadcast it needs is
one shift and one or (`splat` in that file, over `raw` bits) rather than a
conversion. A name with no asking site does not come in.

## `-iquote`, never `-I`, and this one is silent to get wrong

Under NVRTC there is one resolver and the carried set answers both `"…"` and
`<…>`, which is what makes these files work at all. Under **nvcc** there are
two, and the difference is load-bearing:

* `-I shim` would put `cuda_fp16.h` ahead of the **real** toolkit header
  for every angled include in the translation unit. It compiles. `__half`
  becomes `f16`, the mangled symbols change, and the two objects
  measure 17,744 B against 15,088 B with **no diagnostic** —
  `new-horizon.md` §21.10 has the measurement.
* `-iquote shim` answers only `#include "…"`, so no shim can shadow a
  real header no matter what is added here later.

nvcc rejects `-iquote` outright (`nvcc fatal: Unknown option '-iquote'`), so
every site spells it `-Xcompiler=-iquote,<dir>`.

**Two directories, not one.** The traffic crosses the `src`/`shim` seam in
both directions and every one of these five directives used to resolve beside
its includer:

| direction | directive                                              |
| --------- | ------------------------------------------------------ |
| out       | `kernels/prelude/fp8.cuh` → `"cuda_fp16.h"`, `"cuda_fp8.h"` |
| out       | `kernels/prelude/half2.cuh` → `"cuda_fp16.h"`           |
| in        | `shim/cuda_fp16.h` → `"pie_device.cuh"`                 |
| in        | `shim/cuda_bf16.h` → `"pie_device.cuh"`                 |

The inward direction fails loudly if a site forgets a flag — there is no other
`pie_device.cuh` anywhere. **The outward direction does not fail at all**; it
finds the toolkit's header and builds the wrong type. That asymmetry is the
whole reason both directories were named at all four sites:

1. the archive crate's `kernels-cuda/csrc/CMakeLists.txt` —
   `target_compile_options(pie_kernels_cuda …)`
2. `driver-cuda/build.rs` — the `pie_vision_towers` `cc::Build`
3. `driver-cuda/build.rs` — the `pie_attn_flashinfer` `cc::Build`
4. `kernels-cuda/tests/device_typecheck_types.rs` — `compile()`

**All four are gone, and this whole section is now history rather than a
rule.** Site 1 went with the archive crate at `85c6c674b`; sites 2 and 3 went
when `driver-cuda/build.rs` lost every `cc::Build` — what is left of that
script is the link closure behind the `abi` feature and nothing else; site 4's
test file was deleted at `1a08b179a`. `grep -rn iquote crates/` now finds only
prose: five `.cuh` header comments describing a build that no longer runs,
this page and `MODIFICATIONS`. No build script, no
CMakeLists, no test passes the flag, because there is no offline compile of
this crate's device text left to pass it to. So no shim can shadow a real
header today — not because the discipline held, but because the configuration
it protected is gone.

It is kept in full because the hazard is a property of `-I` and not of any
particular build script, and the next offline compile of this text — a
`cc::Build`, a CMake target, a probe that shells out to nvcc — reintroduces it
on the first line. §21.10's measurement is the thing to reread before writing
that flag, not after.

One edge stays inside this directory and needs no flag: `cuda_bf16.h` includes
`"cuda_fp16.h"` and both are here, so it resolves beside the includer. That
one still holds, because it is about the carried set and not about nvcc.

## Under the carried set there is no order, and that is the stronger property

The section above is about **nvcc**, where a shim can shadow a real header.
The question CUTLASS forced is the mirror image of it, and it has to be
written down because the hazard does not survive the translation intact.

Probing CUTLASS with a simulated recipe — `-I kernels -I shim -I csrc/vendor
-I /usr/local/cuda/include` — the **order matters and shim-first loses**:
putting `-I shim` ahead of the toolkit answers `<cuda_fp16.h>`
with ours while CCCL has already pulled the toolkit's, and NVRTC reports
`shim/cuda_fp16.h(236): invalid redeclaration of type name "__half"` followed
by 83 cascading errors. Shim **last** compiles. FlashInfer never showed this
because it does not reach for `<cuda/std/…>`; CuTe does, on nearly every line.

**Under the carried set that hazard cannot occur, because there is no order
and no second answer.** `runtime::nvrtc::options` passes no `-I` at all;
headers are `include_str!`-ed into the binary and handed to
`nvrtcCreateProgram` as `includeNames[]`, matched by literal string. There is
no toolkit behind the shim to be shadowed by it or to shadow it. A name
resolves here or the compile stops.

That inverts the failure mode, and the inversion is worth stating in both
directions:

* **You cannot shadow.** §21.10's defect — a shim silently winning over a real
  header, `__half` becoming `f16`, two objects at 17,744 B and 15,088 B
  and no diagnostic — is structurally impossible under NVRTC. There is nothing
  to win over.
* **You cannot fall through.** An *incomplete* shim is a hard error naming the
  include or the identifier, not a silent wrong build. That is the trade and
  it is a good one: a missing name costs one compile, a wrong name costs a
  bisect.

The residual hazard is neither of those; it is **the probe**, and it caught me
in the same hour I wrote this paragraph. A `-I` probe can fall through to
`/usr/local/cuda/include` and pass on a header the carried set does not have.
Measured, on the two `griddepcontrol` wrappers added below:

| probe | shim `-I` last | shim `-I` first |
| --- | --- | --- |
| `compute_89` (PDL guarded out) | rc=0, 581 B PTX | rc=0, 581 B PTX |
| `compute_90a` (PDL live) | **rc=6**, *"`cudaGridDependencySynchronize` is undefined"* | rc=0, 719 B PTX |

Three of those four cells are green and the shim is only actually being read
in two of them. With `-I` last the toolkit's real `cuda_runtime.h` answers and
the additions are invisible; at `compute_89` the arch guard removes the calls,
so the omission does not show. **A green probe is not evidence the shim was
consulted.** Under the carried set there is no toolkit and no order, so the
shim always answers — which means the JIT is the configuration the right-hand
column measures, and a probe must be arranged to match it.

Nothing catches this later either. The check that walks the carried set is
`src/source.rs::every_device_include_resolves` — it was
`tests/layers.rs::every_include_reachable_from_a_unit_resolves` until
`1a08b179a` deleted that file, and the property is unchanged because both run
on `source::quoted_includes`, which walks **quoted** includes only. Every
CUTLASS and CuTe include is angle-bracketed. Such an omission passes the gate
and fails at first fire on a GPU box, naming the include rather than the
omission. So: an `-I` probe measures
whether the *text* compiles, never whether the *set* is complete. The set is
complete when `src/source.rs` lists the whole closure, and a probe cannot
stand in for that.

## Added for CUTLASS, 2026-08-14 (`new-horizon.md` §62.7)

Left explicit because `xqa-nvrtc` is enumerating shim gaps at the same time
(`cassert`, `__half2`'s two-`__half` constructor, `__hadd2_rn`, `float2` →
`__nv_fp8x2_storage_t`) and two additive commits are better than one
negotiated one. Nothing below removes or rewrites an existing entry.

| file             | added                                              | the site that asked                                                                 |
| ---------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `type_traits`    | `std::is_pointer` / `is_pointer_v`                 | `cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp:497`        |
| `type_traits`    | `std::max`                                         | `cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp:242`       |
| `cuda_runtime.h` | `cudaGridDependencySynchronize()`                  | eleven of CUTLASS's fused-MoE `__global__`s, first statement                          |
| `cuda_runtime.h` | `cudaTriggerProgrammaticLaunchCompletion()`        | the same eleven, last statement                                                       |
| `cuda_runtime_api.h` | the whole file (forwards to `cuda_runtime.h`)  | `cute/util/debug.hpp:38`, reached from `cute/tensor.hpp` — the first line of every CUTLASS unit |

`cuda_runtime_api.h` was added a turn later than the four above, by
`moe-cutlass`, and its own header comment carries the argument for forwarding
rather than declaring. It is worth recording *how* it was missed for so long,
because the answer is not "nobody included it":

**every CUTLASS probe in this project until `G3` ran with
`/usr/local/cuda/include` ahead of `shim/`.** C1b, C4, C5, C6, C13, C14,
T2–T4, E1–E4 — all of them resolved this header, and `cuda_fp16.h`, and
everything else, out of the *toolkit*. That is §62.10 for the third time and
its sharpest instance: the include order under which the CUTLASS probes pass
is an order production cannot enter, so a green probe said nothing about
whether the carried set was complete.

The same probe found a gap that is **not** additive and is therefore not
closed here: CCCL's `extended_data_types.h:50` forward-declares
`struct __half;` under `__has_include(<cuda_fp16.h>)`, and this shim's
`cuda_fp16.h:236` makes `__half` an *alias* to the one canonical device type.
An alias and a struct forward-declaration of one name cannot coexist, and in
production the shim **is** `<cuda_fp16.h>`. Options and the reason none was
taken unilaterally were recorded at
`driver-cuda::fire::flashinfer_moe::params::CCCL_FP16_CONFLICTS_WITH_SHIM_ALIAS`
— it touches the canonical-type identity `fa2` and `xqa` are built on, so it
is a decision rather than an addition. **That symbol no longer exists**:
`efaad26b4` retired the fused CUTLASS leg and deleted the module holding it,
66 files. The argument survives in `new-horizon.md` §68, which is where it was
written out in full and where it is recorded as *deferred, not open*. The gap
itself is unchanged — `cuda_fp16.h:236` is still an alias, CCCL still
forward-declares a struct — and it goes live again the moment anything here
carries CCCL.

`std::void_t` was on CUTLASS's list too
(`epilogue/thread/linear_combination_bias_elementwise.h:77`) and was **already
here** for FlashInfer's `DEFINE_HAS_MEMBER`, so the measured price of three
names was a bill for two.

`std::min` is still deliberately absent and the note in `type_traits` says why
— 58 host-dispatch sites, all guarded away. `max` is here on a different fact
entirely, a `constexpr static` class-scope initialiser, and the two must not
be read as one decision.

## What is *not* here

`pie_device.cuh`, `pie_fp8.cuh`, `pie_half2.cuh`, `pie_mma.cuh` — PIE's own
device text under PIE's own names. Those are not impersonating anything; they
are in `kernels/`. A file belongs here when **the name is somebody else's**.

## Retired with the CUTLASS MoE, 2026-08-14 (`new-horizon.md` §71, §72)

**Seven files were added here and deleted the same day, and the deletion is the
result rather than a retraction.** `cutlass/{array.h, functional.h,
numeric_types.h, numeric_conversion.h, epilogue/thread/activation.h}`,
`cutlass_extensions/epilogue/thread/fused_activations.h` and
`cute/util/type_traits.hpp` answered nine device-side CUTLASS names for
`csrc/src/moe/moe_glue.cuh`, FlashInfer's six MoE glue kernels extracted for
NVRTC. Measured against the carried set with the shim first: **`rc=0`,
142,701 bytes of PTX, seven `.weak .entry`** — the extraction worked.

Then the fused CUTLASS MoE was retired outright, its traffic went to the
general (aligned) path, and the general path has its own glue kernels
(`gather_moe_aligned_inputs`, `chunked_swiglu`, `reorder_moe_aligned_output`,
`token_batched_weighted_sum_add`) which are already rowed and already fire.
`moe_glue.cuh` lost its only consumer, and these seven files lost theirs.

**They are deleted rather than kept for a future caller,** because of the
hazard stated below and still true: the carried set is one flat namespace
searched ahead of everything else, so a partial `cutlass/array.h` found first
is strictly worse than no `cutlass/array.h` at all. A file here is a *floor*,
and a floor with nothing standing on it is a trapdoor.

`cstddef`'s `offsetof` went with them for the same reason — `moe_glue.cuh`'s
ten `static_assert`s were its only asking site. The measurement behind it is
kept below, because unlike the other seven **it is a fact about NVRTC, not
about CUTLASS**, and the next carried header that restates an upstream struct
will need it.

Several rows in the two older sections keep their entries and lose their
asking sites: `type_traits`'s `std::min` (asked by
`finalizeMoeRoutingKernel:1745`), `std::enable_if`/`is_integral` (`cudaUtils.h`'s
`ceilDiv`), `std::is_pointer`/`is_pointer_v` and `std::max` (CUTLASS's two sm90
/sm100 array epilogues), `limits`'s `numeric_limits<float>::infinity()` (the
activation adaptors), and `cuda_runtime_api.h`'s whole forwarding file
(`cute/util/debug.hpp:38`). They are **kept**, because each is a plain
standard-library or toolkit name that any carried header may reach for, and
unlike a `cutlass/` path none of them can shadow a real header with a partial
one.

`cuda_runtime.h`'s two PDL intrinsics are **not** in that category: they have
live askers across the carried attention and MoE kernels and are unaffected.

### The measurement that outlives the row: `offsetof` under NVRTC

`__INTADDR__` is the only spelling that works. Six were tried (probe
`nvrtc-probes/cutlass_moe_832_c22_offsetof.py`):

| spelling | rc | diagnostic |
| --- | --- | --- |
| `__builtin_offsetof(S, d)` | 6 | *type name is not allowed* |
| `offsetof(S, d)` unprefixed | 6 | *type name is not allowed* |
| `(unsigned long)(&((S*)0)->d)` | 6 | *must have a constant value* |
| `(char*)&((S*)0)->d - (char*)(S*)0` | 6 | *must have a constant value* |
| `(char*)&s_.d - (char*)&s_`, `extern S s_` | 6 | *must have a constant value* |
| `__INTADDR__(&((S*)0)->d)` | **0** | — |

NVRTC's front end is EDG; `__INTADDR__` is its intrinsic. **The row worth
keeping is the fifth-and-third:** those pointer-difference spellings are
exactly what every offset probe in `nvrtc-probes/` is built on, and they work
*there* — because a `__constant__` initialiser is folded to `.b8` bytes rather
than evaluated as a constant expression. The same text is legal in one
position and rejected in the other, so an instrument that works is not
evidence that the construct is legal.

### The findings that outlive the seven files

Both are in `new-horizon.md` §72 in full; the short forms, because this is
where someone re-adding a `cutlass/` shim would stand:

**`cutlass::Array<bf16, N>` is a bit-packed proxy container, not `T[N]`.**
`Array<T, N, RegisterSized = sizeof_bits<T>::value >= 32>` sends every 16-bit
element type to `array_subbyte.h`'s specialisation, whose `Storage` is
`unsigned int`. Measured: `sizeof(Array<bf16,8>) == 16`, **`alignof == 4`**,
where a transcription writing `bf16 storage[8]` gets align 2.

**`PropagateNaN` is load-bearing.** `ReLu` passes `true`, which emits
`max.NaN.f32` rather than `max.f32`. A one-parameter `minimum`/`maximum` is a
silent wrong answer everywhere.

**`GELU_taylor` has a `float` specialisation that is a different function from
the generic.** The PTX difference was published as NVRTC reassociating behind
a folded constant `0f3D122279`, and withdrawn: it is
`float(0.7978845608028654 * 0.044715)`, written out longhand at
`activation.h:652`. A specialisation you did not look for reads exactly like
an optimiser you cannot argue with.

### The shadowing hazard, stated once and now acted on

The carried set is one flat namespace searched ahead of everything else. A
file here named `cutlass/array.h` *is* `cutlass/array.h` for every carried
translation unit. So if a later pass decides to carry real CUTLASS, nothing
partial may be left in its path — which is why the seven are gone rather than
dormant, and why `csrc/vendor/cutlass_extensions/` (one patched file, carried
but unreached) was deleted in the same change. Its two-line frontend fix is
recorded in `new-horizon.md` §72.6, which is the only place it now exists.

## Retired: `cuda/pipeline`, 2026-08-16

**503 lines answering a directive nothing stood behind.** The file implemented
`cuda::pipeline`, `cuda::pipeline_shared_state`, `cuda::memcpy_async` and
`cuda::aligned_size_t` over `cp.async.ca/cg.shared.global`,
`cp.async.commit_group` and `cp.async.wait_group`, because
`flashinfer/permuted_smem.cuh:23` said `#include <cuda/pipeline>` and a
directive the set does not answer stops the compile.

Its own banner had already recorded the finding that retired it. Counted over
the 28-file attention closure, over the FlashInfer tree outside `3rdparty/`,
and over our own device text, every name it defined was reached **zero**
times:

| name | closure | flashinfer tree | our kernels |
|---|---|---|---|
| `#include <cuda/pipeline>` | 1 | 1 | 0 |
| `cuda::pipeline` | 0 | 0 | 0 |
| `cuda::memcpy_async` | 0 | 0 | 0 |
| `cuda::pipeline_shared_state` | 0 | 0 | 0 |
| `cuda::aligned_size_t` | 0 | 0 | 0 |

FlashInfer stages through shared memory constantly and does it with its own
`cp_async.cuh` wrappers — `commit_group()`, `wait_group<n>()`, `load_128b<>()`
— straight onto the instructions this file was emitting. The one place
upstream reaches for a libcu++ synchronisation object, `mamba/
kernel_selective_state_update_stp.cuh`, uses `cuda::barrier` and TMA and is
not in our closure. The include was vestigial, and the shim answered a
vestige.

**The include went with it,** which is why this is a deletion and not a
`#pragma once`. The banner's argument for keeping a working implementation was
that an empty header answers the directive and then answers `cuda::pipeline`
with *"namespace cuda has no member"* at some future call site — a diagnostic
that reads like a missing include and sends the reader after a file that is
already found. That argument is sound and it is why the *file* could not
simply be emptied; it says nothing for keeping the *directive*. With
`permuted_smem.cuh:23` marked `// PIE: REMOVED`, there is no asker, so there
is no name to answer and no void to answer it with.

This is the same rule the CUTLASS section states: **a file here is a floor,
and a floor with nothing standing on it is a trapdoor.** `cuda/pipeline` was
a floor under an empty room.

What made it defensible to keep for as long as it was kept is also gone.
`examples/fp8_pipeline_probe.rs` compiled one staging kernel twice — `nvcc`
against NVIDIA's real `<cuda/pipeline>`, NVRTC against this shim — and
required identical bytes on the same device from the same input. That probe
went with the whole of `kernels-cuda/examples/`, so the implementation had
been unverified as well as unreached. A synchronisation primitive checked only
against itself is worth very little; one checked against nothing, and called
by nothing, is worth its line count in the opposite direction.

`shim/cuda/` keeps `cmath` and `std/limits`, which have live askers
(`fastdiv.cuh:19` and `attention/mla.cuh:21`). The directory stays; only the
one file left.

## Retired: `bit`, `boost/math/ccmath/fabs.hpp`, `limits`, `cuda_runtime_api.h`, 2026-08-16

**Four files that answered directives no NVRTC compile takes, found by
deleting each shim in turn and recompiling.** The probe is the method this
page had been missing: for every file here, drop it from the header set and
run `tests/every_instantiation_compiles.rs`. A shim with a live asker fails
loudly — `catastrophic error: could not open source file "X"`. A shim with
none compiles clean, and there were four.

| file | its asker | why the asker never fires |
|---|---|---|
| `bit` | `flashinfer/fp16.h` | `fp16.h` is included under `#ifdef FP16_QK_REDUCTION_SUPPORTED`, which nothing defines |
| `boost/math/ccmath/fabs.hpp` | `flashinfer/fp16.h` | same |
| `limits` | `xqa/mha_stdheaders.cuh:25` | inside `#ifndef GENERATE_CUBIN`, and every XQA unit sets `GENERATE_CUBIN=1` |
| `cuda_runtime_api.h` | `xqa/barriers.cuh:29`, `xqa/mha.h:20` | inside `#ifndef __CUDACC__`, which NVRTC defines |

`fp16.h` went with the first two — it is upstream's, it is 179 lines of
device text, and no compile has ever reached it. Its deletion is recorded in
`MODIFICATIONS` under **AND A FIFTH FILE WENT**, and the licence consequence
in `NOTICE`: it was the only MIT file this crate redistributed.

### The two that this page had already argued for keeping

`limits` and `cuda_runtime_api.h` are the interesting pair, because the
**Retired with the CUTLASS MoE** section above names both and keeps them:

> They are **kept**, because each is a plain standard-library or toolkit name
> that any carried header may reach for, and unlike a `cutlass/` path none of
> them can shadow a real header with a partial one.

That argument was about *shadowing*, and it is still correct about shadowing.
It was silently doing a second job it was never tested for — standing in for
"and therefore this file earns its place" — and a measurement now says it does
not. Both had lost their asking sites when the CUTLASS MoE was retired. What
this page recorded at the time was that they *could* be kept safely; what it
did not record is that nothing was left asking.

The remaining askers are the ones in the table, and every one of them is
behind a host-only guard. That is a stronger statement than "no asker": these
are directives that exist, that a reader will find with `grep`, and that NVRTC
structurally cannot take. A shim answering one is not dormant — it is
answering a question already asked and answered somewhere else.

`cuda/std/limits` is a different file and **stays**: `attention/mla.cuh:21`
reaches it unguarded, and deleting it breaks the compile. The two are easy to
confuse and the probe told them apart in one run.

### What the probe says about the rest

The other twelve all break when removed, with `cuda_fp16.h` and `cuda_fp8.h`
the only two also reached by a quoted directive. That is the first time this
directory's contents have been justified by anything other than the banner in
each file, and the banners were right about all twelve.
