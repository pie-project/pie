# `spec/` — text that is cited, not compiled

One file, and a rule.

## The rule

`csrc/` holds device text and nothing else. A file that no compiler reads and
that is written in host C++ is neither a kernel nor a shim, and while it sat
under `csrc/src/attn/` it made that rule a thing a reader had to check rather
than one the directory stated. `build.rs`'s `carried` module walks `csrc/` and
carries **every** file it finds into the binary as a header NVRTC can resolve
against, so a host header living there is also 45 KB of `std::vector` in an
`includeNames[]` array — harmless only for as long as nothing includes it.

Anything in this directory is the opposite promise: **no build step reads it,
and its line numbers are a public interface.**

## The file

`attention_flashinfer_common.cuh` is the FA2 archive's specification. It was
the shared body of `attn/attention_flashinfer_hd{64,128,256,512}.cu`, all four
of which are deleted, and it has had **zero `#include` consumers anywhere in
the workspace** since. Measured, not assumed:

```
grep -rnE '^[[:space:]]*#[[:space:]]*include.*attention_flashinfer_common' crates/
```

returns nothing. `src/source.rs`'s `every_include_reachable_from_a_unit_resolves`
already named it as its one exemption for exactly this reason, and the file's
own EOF note item 2 reached the finding independently.

What it still is: the thing twenty-four live citations of the form
`attention_flashinfer_common.cuh:NNN` point into, from
`driver-cuda/src/fire/flashinfer_fa2.rs`, `fire/flashinfer_fa2_dispatch.rs`,
`kernels-cuda-new/src/families/fa2.rs`, `src/fa2.rs`, `csrc/src/attn/fa2.cuh`
and the archive's CMakeLists. The Rust that replaced it was ported *against*
these lines — `DecodePlanCache` is `:341-374`, `PrefillPlanCache` is `:376-400`,
`run_decode`'s params filling is `:581-641` — so the citations are how a reader
checks the port, and they are worth more than the disk the file costs.

## Editing it

**Append at EOF or not at all.** `tests/launch_rules.rs`'s
`the_fa2_specification_has_not_shifted_under_its_citations` pins seven lines
between 79 and 878 and fires if any of them moves. That test is a shift
detector: it cannot tell you a citation is *wrong*, only that every citation
below the edit is *now* wrong. Insert a line at the top and twenty-four
references go stale at once, silently, because no compiler reads this file and
nothing else would notice.

The `git mv` that brought it here changed the path and not one byte above the
EOF note, so every citation still lands.
