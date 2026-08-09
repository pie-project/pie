# `csrc/shim` — headers that answer for a compiler that is not here

Every file in this directory wears somebody else's filename on purpose.

NVRTC before CUDA 13.3 ships **no device headers at all**. A `#include
<cuda_fp16.h>` under it does not find a smaller `cuda_fp16.h`; it finds
nothing, and the compile stops. The device text this crate carries — ours and
FlashInfer's alike — is written against those names, so the names have to be
answered. These fourteen files are the answer.

## The name is the contract

Nothing here may be renamed. NVRTC has no `-I` and no search list: it matches
the **literal string in the `#include` directive** against the names in
`includeNames[]`. A file called `cuda_fp16.h` is answering for NVIDIA's
`cuda_fp16.h` and stops answering the moment it is called anything else.

That is also why moving them here cost nothing. `build.rs`'s `carried` module
names a file by its path relative to the tree it sits in, so `csrc/shim/
cuda_fp16.h` is carried as `cuda_fp16.h` exactly as `csrc/src/cuda_fp16.h`
was. **A compile cannot tell this move happened.** It is the one role in
`csrc/` with that property, which is why it moved first.

## Why they are together, and what they were before

They used to be split across two directories by **who wrote them**:

| was                  | files                                                                                                     | why it was there                |
| -------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `csrc/src/` (8)      | `cuda_fp16.h` `cuda_bf16.h` `cuda_fp8.h` `cuda_fp4.h` `cooperative_groups.h` `cuda/cmath` `cuda/pipeline` `cuda/std/limits` | PIE wrote them                   |
| `csrc/vendor/` (6)   | `cstdint` `type_traits` `bit` `cuda.h` `cuda_runtime.h` `boost/math/ccmath/fabs.hpp`                       | they arrived with the vendoring |

Same job, two directories, and the only thing the split recorded was a fact
about history. A compile does not care who wrote a shim; it cares that the
name resolves. Filed by **role**, all fourteen are one thing.

The six on the second row have a second story worth keeping: every one of them
was a `#ifndef __CUDACC_RTC__` guard first, and every one of those guards was
measured and refused. Guarding `#include <cstdint>` compiles and then deletes
`uint32_t` from 2,512 device declarations. Guarding `<cuda_runtime.h>` deletes
`ushort`, which `math.cuh`'s `ex2.approx.f16` wrapper is written in. Guarding
`<cuda.h>` silently unsets `CUDA_VERSION` and the fp4 vector types disappear
with no diagnostic at all. A host header whose names reach device code is
**carried**, under the exact spelling the directive uses — the rule that is
the difference between 33 guards in the vendored tree and roughly seventy.

## `-iquote`, never `-I`, and this one is silent to get wrong

Under NVRTC there is one resolver and the carried set answers both `"…"` and
`<…>`, which is what makes these files work at all. Under **nvcc** there are
two, and the difference is load-bearing:

* `-I csrc/shim` would put `cuda_fp16.h` ahead of the **real** toolkit header
  for every angled include in the translation unit. It compiles. `__half`
  becomes `device::f16`, the mangled symbols change, and the two objects
  measure 17,744 B against 15,088 B with **no diagnostic** —
  `new-horizon.md` §21.10 has the measurement.
* `-iquote csrc/shim` answers only `#include "…"`, so no shim can shadow a
  real header no matter what is added here later.

nvcc rejects `-iquote` outright (`nvcc fatal: Unknown option '-iquote'`), so
every site spells it `-Xcompiler=-iquote,<dir>`.

**Two directories, not one.** The traffic crosses the `src`/`shim` seam in
both directions and every one of these five directives used to resolve beside
its includer:

| direction | directive                                              |
| --------- | ------------------------------------------------------ |
| out       | `csrc/src/pie_fp8.cuh` → `"cuda_fp16.h"`, `"cuda_fp8.h"` |
| out       | `csrc/src/pie_half2.cuh` → `"cuda_fp16.h"`             |
| in        | `shim/cuda_fp16.h` → `"pie_device.cuh"`                 |
| in        | `shim/cuda_bf16.h` → `"pie_device.cuh"`                 |

The inward direction fails loudly if a site forgets a flag — there is no other
`pie_device.cuh` anywhere. **The outward direction does not fail at all**; it
finds the toolkit's header and builds the wrong type. That asymmetry is the
whole reason both directories are named at all four sites:

1. `kernels-cuda/csrc/CMakeLists.txt` — `target_compile_options(pie_kernels_cuda …)`
2. `driver-cuda/build.rs` — the `pie_vision_towers` `cc::Build`
3. `driver-cuda/build.rs` — the `pie_attn_flashinfer` `cc::Build`
4. `kernels-cuda-new/tests/device_typecheck_types.rs` — `compile()`

One edge stays inside this directory and needs no flag: `cuda_bf16.h` includes
`"cuda_fp16.h"` and both are here, so it resolves beside the includer.

## Under the carried set there is no order, and that is the stronger property

The section above is about **nvcc**, where a shim can shadow a real header.
The question CUTLASS forced is the mirror image of it, and it has to be
written down because the hazard does not survive the translation intact.

Probing CUTLASS with a simulated recipe — `-I csrc/src -I csrc/shim -I
csrc/vendor -I /usr/local/cuda/include` — the **order matters and shim-first
loses**: putting `-I csrc/shim` ahead of the toolkit answers `<cuda_fp16.h>`
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
  header, `__half` becoming `device::f16`, two objects at 17,744 B and 15,088 B
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

Nothing catches this later either:
`tests/layers.rs::every_include_reachable_from_a_unit_resolves` walks
**quoted** includes only, and every CUTLASS and CuTe include is angle-
bracketed. Such an omission passes the gate and fails at first fire on a GPU
box, naming the include rather than the omission. So: an `-I` probe measures
whether the *text* compiles, never whether the *set* is complete. The set is
complete when `carried.rs`'s walk covers the closure — the directory IS the
set — and a probe cannot stand in for that.

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
are in `csrc/src`. A file belongs here when **the name is somebody else's**.
