# What deleting `bridge` costs — measured, not estimated

Measured on branch `rewrite`, read-only. Nothing here was built, checked or
run; every row is a citation to a file and line. The method is stated at the
bottom so a disagreement can be reproduced rather than argued.

**The number that decides the schedule: 14 `bridge`-gated items have a caller
that is not `bridge`-gated, across 69 call sites in 5 files.** One of those
files is broken *today*; the other four break at the moment `abi` stops
implying `bridge`.

**Step 6 is an afternoon, not a week** — because the 69 sites are 5 files, not
5 subsystems, and 35 of the 69 are in one file (`fire/launch.rs`) that dies
with the feature anyway once `serve` stops calling it.

> **⚠ Read §2.1, §2.2 AND §2.3 before doing any of this.** §2.1: three rows of
> §2's table say an item "dies" and that nothing notices; they are wrong, and
> `bind::facts`, `dispatch` and `window_of` must outlive the feature. §2.2:
> the fix §2.1 prescribes is *also* wrong, because most of this gate is
> inherited rather than earned. §2.3: the census is **76 attributes across 13
> files**, not 36 across 10 — and one more item on the "dies" list is a
> homonym whose deletion takes the graph-replay path with it.
>
> **Measured verdict (`e88f1ffff`): 1 earns the gate · 3 die with the row
> world · 4 are tests · 68 are fossils that RE-GATE `_cuda`.** The headline
> below is not a measure of work; it is a measure of breakage that re-gating
> makes not happen.

---

## 0. The feature graph, as declared

| key | value | where |
|---|---|---|
| `bridge` | `["kernels-cuda/native"]` | `driver-cuda/Cargo.toml:~100` |
| `abi` | `["bridge", "dep:driver-api", "dep:model-loader", "model/contract"]` | same |
| `jit-parity` | `[]` — empty list, doc says "Needs bridge" | `:103` |
| `_cuda` | `["dep:cudarc"]`, set by `cuda-12` / `cuda-13` | same |
| `native` | `["dep:cc"]` (`dep:cmake` already gone) | `kernels-cuda/Cargo.toml:29` |
| `links` | `"pie_kernels_cuda"` | `kernels-cuda/Cargo.toml:21` |

There is **deliberately no `default`**, and `_cuda` is what gates the modules.
`bridge` is *not* a superset of `_cuda`: `cargo test -p driver-cuda --features
cuda-12` (`.github/workflows/ci.yml:333`) is `_cuda` **on**, `bridge` **off**,
and it is a job that is supposed to pass.

### The single structural fact everything below follows from

`abi = ["bridge", …]`. So today, *anything gated on `abi` is transitively
gated on `bridge`*, and the compiler cannot tell the two apart. **Step 6 breaks
that implication** — `abi` keeps `driver-api`, `model-loader` and
`model/contract` and loses `bridge`. Every `abi`-gated reference to a
`bridge`-gated item is a compile error created by that one edit to
`Cargo.toml`, and none of them is visible before it.

---

## 1. The gates themselves

**60 attribute sites** in `driver-cuda/src` name `bridge` or `jit-parity`
(59 `#[cfg(…)]` on items, 1 `#![cfg(…)]` on a module), covering **36 named
items**. `kernels-cuda/build.rs` adds 2 `#[cfg(feature = "native")]`.

**Zero `#[cfg(not(feature = "bridge"))]` exist anywhere in the tree.** Not one
of the 36 items has a written post-`bridge` form, and no `cfg!(feature =
"bridge")` runtime branch exists either. That is a clean result, not a gap:
the retirement is a deletion, not a migration, and there is no second
implementation to reconcile.

### Module tiers — the reachability model

| tier | gate | modules | step 6 |
|---|---|---|---|
| **A** | `bridge` | `tower` + `gemma4_audio`, `gemma4_vision`, `qwen3_vl`, `qwen3_vl::attn` (`lib.rs:226`); `bind::facts` (`bind/facts.rs:44`, `#![cfg]`) | **deleted whole** |
| **B** | `abi` | `fire::launch` (`fire/mod.rs`), `serve` + `encode`/`load`/`state`/`transfer` (`lib.rs:230`), `weights::plan`, `weights::stage` (`weights/mod.rs`) | **survives — and is where the errors land** |
| **C** | `_cuda` only | `bind` (incl. `service`, `abi`, `jit`, `nvrtc`, `launch`), `device` (incl. `graph`), `pools` (incl. `kv_cache_live`), `fire` (all but `launch`), `program`, `weights::weight_view` | **survives; holds bridge-gated items inside it** |

Tier C is the interesting one: `bind`, `device`, `pools` and `fire` are
compiled by the `--features cuda-12` job, and each of them *contains*
`bridge`-gated items. The gate is inside the module, not on it.

---

## 2. Items that die with the feature — no home needed

| item | where | dies or moves | what notices if this is wrong |
|---|---|---|---|
| `DecodePlan`, `PrefillPlan`, `PrefillPlanFlags` + their `Default`/`Drop`/`impl` | `bind/mod.rs:698,704,838,845,856,862,1043,1059,1066` | **dies** | `serve/state.rs` field types — tier B, §4 |
| `DispatchCtx` + `impl` | `bind/mod.rs:1112,1316` | **dies** | `bind/service.rs` — tier C, §4, **broken today** |
| `AttnCtx` | `bind/mod.rs:1355` | **dies** | `fire/launch.rs` — tier B |
| `GdnCtx` | `bind/mod.rs:1472` | **dies** | `fire/launch.rs` |
| `DispatchRefusal`, `RunRefusal`, `RunRefusalKind` | `bind/mod.rs:1521,2785,2797` | **dies** | `fire/launch.rs:224` return type |
| `AttnRegions` + `impl` | `bind/mod.rs:2820,2830` | **dies** | `fire/launch.rs:216,3570,3571` |
| `dispatch_generated`, `dispatch`, `run`, `run_captured` | `bind/mod.rs:1601,2225,2868,2963` | three **die**; **`dispatch` SPLITS** | `fire/launch.rs` — 10 call sites — **and, unlisted until now, `dispatch` holds the tree's only `Cx::new` (`:2379`). See §2.1.** |
| `Arms`, `dispatch_jit_probe` | `bind/mod.rs:2153,2189` | **dies** | nothing outside the gate |
| ~~`window_of`~~, `isqrt_exact`, `stage_d2d`, ~~`cond_path`~~, `siblings`, ~~`arm_body`~~, ~~`OpenCond`~~ | `bind/mod.rs:238,2721,2731,2913,2934,3085,2905` | three **die**; **`window_of` MOVES**; **`cond_path`/`arm_body`/`OpenCond` are GRAPH machinery** | ~~nothing — all callers are inside `dispatch*`/`run*`~~ — **false twice.** §2.1: `bind/facts.rs:234` calls `window_of`. §2.3: `arm_body(cond: &device::Cond, conds: &[lower::CondRegion])` and `OpenCond { cond: device::Cond }` are **CUDA-graph conditional** arms, not dispatch `match` arms — deleting them deletes `run_captured`. |
| `abi::ffi` module + `seed_envelopes_empty` | `bind/abi.rs:318,351` | **dies** | it is the `include!` host, §5 |
| ~~`bind::facts` — whole file~~ | `bind/facts.rs:44` | **MOVES — it must outlive the feature** | ~~nothing; `#![cfg]` makes the file self-gating~~ — **false. `impl Facts for Fire<'_>` (`:117`) is the ONLY `impl Facts` in the tree.** Every fn-world body notices. See §2.1. |
| `CaptureInfo`, `capture_info` ×2, `update_capture_deps` ×2, `Cond`, `Switch`, `SupergraphBuilder` + `impl`s, the `cudarc::runtime::sys` import | `device/graph.rs:624,626,813,828,857,888,904,926,934,954,961,994,1014` | **dies** | `device/mod.rs:64` re-export; `fire/launch.rs:325` |
| `device::{Cond, SupergraphBuilder}` re-export | `device/mod.rs:64` | **dies** | `fire/launch.rs:325` |
| `LiveLoraOps` + 3 `impl`s, `lora::apply` | `fire/lora.rs:177,183,198,217,730` | **dies** | `fire/launch.rs:2543` |
| `capture_digest` | `fire/recordings.rs:684` | **dies** | `fire/launch.rs:271` |
| `LiveKvCacheOps` + 2 `impl`s | `pools/kv_cache_live.rs:82,90,137` | **dies** | `fire/launch.rs:1560`, `serve/transfer.rs:565` |
| `tower` module tree | `lib.rs:226` | **dies** | `serve/encode.rs:29,30` — tier B |

**Nothing in this list needs a home** — *except the three rows struck through
above, and the exception is §2.1.* Everything else is a host-side program for
the C++ archive, and the fn-world successor is already elsewhere
(`x/**`, `fire/*` non-`launch`). The `impl` blocks are counted separately from
their types because each carries its own attribute.

---

## 2.1 The correction: the row world and the fn world are gated on the same feature

**Re-derived this session, read-only. Three rows of the table above are wrong,
and all three are wrong the same way.**

The chain, six links, each read:

1. `bind::resolve` classifies every symbol through `x::route` — **ungated**
   (`bind/mod.rs:331`).
2. The one arm that *executes* a crossed symbol is
   `Route::Bound(entry) => entry.call(&Cx::new(&fire), ctx.stream)`
   (`bind/mod.rs:2327, 2379`).
3. It sits inside `pub fn dispatch<R: Resolver>` — `#[cfg(feature = "bridge")]`
   (`bind/mod.rs:2254`).
4. `Cx::new(facts: &dyn Facts)` is `Cx`'s **only** constructor
   (`kernels-cuda-new/src/x/cx.rs:768`), and `bind/mod.rs:2379` is its **only**
   caller, tree-wide.
5. Its argument is `facts::Fire`; `impl Facts for Fire<'_>`
   (`bind/facts.rs:117`) is the **only** `impl Facts` in the tree, in a file
   that is `#![cfg(feature = "bridge")]` (`:44`).
6. `Cx::window_left` (`bind/facts.rs:234`) calls `super::window_of`, which the
   table above says dies.

> **After step 6 as written, the driver can still classify every symbol as
> `Route::Bound`, and can no longer fire one.** Classification survives the
> feature; execution does not.

### Why it fails silently — twice

`cargo test -p driver-cuda --features cuda-12` (`ci.yml:333`) is `_cuda` **on**,
`bridge` **off**, and is supposed to pass. Under it every fn-world body
compiles and nothing in the crate can construct a `Cx`. **Unreachable code is
not a compile error.**

And `Facts` is **fully defaulted** — 52 methods, 46 returning `None`. So
`impl Facts for X {}` compiles, and all 44 `Cx` queries become
`Refusal::Unstated { what: … }`: a message naming the **fact** and never the
**cause**. A reader gets *"nothing states a sm_scale"* when the truth is
*"the driver has no facts at all."*

### The fix — the shape the tree already has a name for

**The body moves to the surviving side**, as `dequant_kv_cache_layer_to_bf16_active`,
`attn_plan_for` and the six FA2 entry points all did. `bridge` is
`["kernels-cuda/native"]` / `links = "pie_kernels_cuda"`; nothing in `Fire`,
`impl Facts`, `window_of` or the `Route::Bound` arm needs a symbol from that
archive.

```
Route::Bound(entry)     -> _cuda    the fn world; it must outlive the feature
Route::Rows             -> bridge   the generated shim; it dies with it
Route::Unbound/Unknown  -> _cuda    load-time refusals naming a gate, not a body
bind/facts.rs           -> _cuda    drop the `#![cfg(feature = "bridge")]`
window_of               -> _cuda    `Cx::window_left`'s body
```

### What it costs the headline

**Nothing, in count.** 69 sites, 5 files, 35 of them in `fire/launch.rs` which
still dies. Step 6 is still an afternoon. What changed is that the afternoon
has a **first** step, and this table asserted it was unnecessary.

### How it was found, and the rule it earns

Not by building — by checking this document against the tree it describes.
When the table was written, `bind::facts` served the tower and `Cx` had a
handful of queries; it now carries 44 and ten crossed families. **The claim
was derived once, from a set that then changed underneath it.**

> A row that says **"nothing notices"** is the most expensive claim a ledger
> can hold, because it is the one row nobody re-derives — its whole content is
> a promise that there is nothing to look at.

Every other row here names a file and a line, so a doubter has somewhere to go.
Three rows said "nothing". **Re-deriving all three found one wrong outright
and one falsified since — and the two that survived, `Arms` and `siblings`,
matched only the English words in prose.** The count was never the finding;
reading the hits was.

---

## 2.2 The correction to the correction: 36 items carry the gate, one earns it

**§2.1's prescribed fix — "`bind/facts.rs` → `_cuda`, drop the `#![cfg]`" — is
wrong,** and it is wrong the way §2.1's own rule predicts: derived once, from
one item, never run across the set.

`facts::Fire` has nine fields. Three — `ctx: &DispatchCtx`,
`attn: Option<&AttnCtx>`, `gdn: Option<&GdnCtx>` — are **`bridge`-gated types
on the "dies" list above**. Dropping `Fire`'s gate while its field types keep
theirs is an ungated item whose type is gated: **exactly the `f38d199c2` break,
recreated by the fix meant to prevent it.**

So the question is not where the gate goes. It is what the gate is *for*.

### The measurement

`bridge = ["kernels-cuda/native"]`, `links = "pie_kernels_cuda"`. The archive
is reached from **one** place in `driver-cuda/src`:

```rust
// bind/abi.rs:318
#[cfg(feature = "bridge")]
pub mod ffi { include!(concat!(env!("OUT_DIR"), "/launch_bindings.rs")); }
```

**36 gated items across 10 files. One reaches it.** Every other match for a
call into the archive is `std::ffi` — the standard library module, a
**homonym** — or a comment about a `pie_k_*` that no longer exists.

### The specimen

`bind::abi::seed_envelopes_empty` is `bridge`-gated and listed as **dies**. Its
doc heading: *"This was the tree's
`ffi::pie_k_layout_launch_envelope_seed_empty_bf16`"*. Its body calls
`kernels_cuda_new::x::layout::envelope_seed_empty` — **fn-world**. It was
crossed, the dependency went, the attribute stayed. Deleting it deletes a
crossed kernel's only caller.

> A `bridge` gate marks where the archive **was**, not where it **is**. This
> document classified items by their attribute, so it is a census of history.

### Three classes; §2's table has one column

| class | test | items | step 6 action |
|---|---|---|---|
| **earns it** | reaches `launch_bindings.rs` | `bind::abi::ffi` — **one** | delete |
| **dies with the ROW WORLD** | interprets `KernelSig` rows | `dispatch_generated`, `Arms`, `arm_body`, `OpenCond`, … | delete — reason is `ROW_TABLES`, **not the archive** |
| **fossil** | body is fn-world or plain data | `seed_envelopes_empty`, `DispatchCtx`, `AttnCtx`, `GdnCtx`, `facts::Fire`, `window_of`, the `Route::Bound` arm | **re-gate `_cuda`** — deleting them deletes live code |

`DispatchCtx` decides its own class: `stream`, `cublas`, `eps`, `rope_theta`,
head counts, `token_ids`, `positions`, `final_logit_softcap`, `ple_dim`, two
`Vec`s. **Scalars, vectors and device pointers — not one archive type.** It is
the fire's model-wide facts; `bridge` is a C++ archive. The gate was applied to
the **consumer** and inherited by the **data**.

### The discriminator

> **A `bridge` item is one whose body needs a symbol from the C++ archive.
> Name the symbol. If you cannot, the gate is inherited or fossil, not
> earned.**

### It convicts this document's own author

`f38d199c2` fixed a live break — `bind::service` ungated, naming `DispatchCtx`
27 times — **by gating `service` to `bridge`.** If `DispatchCtx` is a fossil,
and its fields say it is, the correct fix was the opposite: retract the gate
from `DispatchCtx` to `_cuda`. **Propagating a gate outward from a type to its
consumer is exactly the mechanism that produced the other 34.** The fix was
right that the two must agree and wrong about which was lying, and it made this
census one item worse while reporting itself as a repair.

### What step 6 actually is

1. delete `bind::abi::ffi`, `launch_bindings.rs`, `emit_rust_bindings`, and the
   shim linkage in `driver-cuda/build.rs`
2. delete the row-world interpreters — gated on `ROW_TABLES` emptying, which is
   `attn`'s remaining rows and nothing else
3. **re-gate the fossils `bridge` → `_cuda`** — nothing moves, nothing is
   ported, no body is rewritten

**§4's headline is not a measure of work. It is a measure of breakage that
step (3) makes not happen.** `fire/launch.rs`'s 35 sites never reached the
archive.

---

## 2.3 The census, corrected a third time — and `arm` is a homonym

**`e88f1ffff`, measured by anchoring on `#[cfg(` / `#![cfg(` rather than on the
word `bridge`** — which also appears in ~40 comments and three Cargo stanzas.

| | §2.2 said | measured |
|---|---|---|
| gate attributes | 36 across 10 files | **76 across 13** |
| earns the gate | 1 | **1** — `bind::abi::ffi` |
| dies with the ROW WORLD | — | **3** — `dispatch_generated`, `Arms`, `dispatch_jit_probe` |
| tests | — | **4** |
| **fossil → re-gate `_cuda`** | "most" | **68** |

The archive search returned **16** `ffi::`-shaped tokens before it returned
**1** live one — fifteen were `std::ffi`, `core::ffi`, or comments about dead
`pie_k_*` symbols. The extractor proved it could find something before its
answer was believed, which is §79's rule 3.

### The third dangerous row, and it is on §2.1's own page

§2's table lists as dying-and-unnoticed: `window_of`, `isqrt_exact`,
`stage_d2d`, `cond_path`, `siblings`, `arm_body`, `OpenCond`.

§2.1 re-derived that row, caught `window_of`, and cleared `siblings` and `Arms`
**because every hit outside the gate was the English word in prose** — writing
the homonym trap down, in those words, on that page. And letting `arm_body`,
`OpenCond` and `cond_path` through unchecked.

```rust
fn arm_body(cond: &crate::device::Cond, conds: &[model_compiler::lower::CondRegion], …)
struct OpenCond { cond: crate::device::Cond, node }
let mut stack: Vec<OpenCond> = Vec::new();      // :3799, used :3838 :3852 :3854
```

An **arm of a CUDA-graph conditional**, not of a dispatch `match`. Deleting the
three as this table directs deletes `run_captured` — the graph-replay path.

### The rule

The first classifier reproduced the error before catching it, because its regex
was seeded with the names in this table: it matched `arm_body` against
`arm_body` and reported agreement.

> **A classifier seeded from the claim it checks returns the claim.**

Different from *"read the hits, not the count"* — here every hit is real. The
defect is upstream of the matching: the **question** came from the thing under
test. Derive the candidate set independently of the claim, or the derivation
cannot disagree.

### Two decisions in the re-gate that are not about counting

**`bridge = ["_cuda", …]`.** `bridge` alone never compiled — `device/graph.rs`
names `cudarc::runtime::sys` 27 times behind it. Stating the dependency adds no
configuration that ever built; it adds the **order** that lets 68 retractions
land ahead of 4 deletions without an intermediate state failing to compile for
a new reason.

**`dispatch` was split by statement, not by function.** `Route::Bound` is
`Cx::new`'s only caller and must outlive the feature; its `dispatch_generated`
call must not. The call is **gated, not deleted**: `Route::Rows` is still
constructible (`x/mod.rs:388`), and whether such a symbol is also a `DriverOp`
is a registry fact only a build answers.

---

## 3. The `abi`-gated modules — what they lose

**Read §2.2 and §2.3 first: these modules lose far less than this table says,
because the items they name are mostly fossils that re-gate rather than die.**


| module | gate | what it loses | step 6 action |
|---|---|---|---|
| `fire::launch` | `abi` | 35 references to 12 dead items | **dies with `bridge` in practice**: `capture_or_replay`, the `run`/`run_captured` legs and the `DispatchCtx`/`AttnCtx` builders are its body. Not a port. |
| `serve::state` | `abi` | `DecodePlan` ×3, `PrefillPlan` ×1 as **struct fields** (`:299,302,303,314`) | fields removed; every constructor of that struct notices |
| `serve::encode` | `abi` | `crate::tower::{gemma4_audio, gemma4_vision}` (`:29,30`) | the two towers go with `tower`; `encode`'s multimodal legs lose their implementation |
| `serve::transfer` | `abi` | `LiveKvCacheOps` (`:565`) | one construction site |
| `weights::plan`, `weights::stage` | `abi` | nothing | unaffected — they were `abi` for `model-loader`, not for the archive |

---

## 4. **The headline: callers that are not gated**

**14 items · 69 sites · 5 files.**

| tier | file | sites | items referenced | when it breaks |
|---|---|---|---|---|
| **C** | `bind/service.rs` | **27** | `DispatchCtx` | **already broken** |
| B | `fire/launch.rs` | 35 | `AttnCtx`, `AttnRegions`, `DecodePlan`, `DispatchCtx`, `GdnCtx`, `LiveLoraOps`, `PrefillPlan`, `RunRefusal`, `SupergraphBuilder`, `capture_digest`, `run`, `run_captured` | when `abi` drops `bridge` |
| B | `serve/state.rs` | 4 | `DecodePlan`, `PrefillPlan` | same |
| B | `serve/encode.rs` | 2 | `tower` | same |
| B | `serve/transfer.rs` | 1 | `LiveKvCacheOps` | same |

### `bind/service.rs` is a compile error that exists now

`bind/service.rs` is declared **ungated** at `bind/mod.rs:41`, so it is
compiled whenever `_cuda` is on. It names `DispatchCtx` **27 times** — `use
super::DispatchCtx;` at `:72` and 26 function signatures (`:238, 287, 476,
538, 598, 951, 1024, 1100, 1135, 1182, 1221, 1254, 1287, 1319, 1356, 1384,
1419, 1468, 1533, 1569, 1702, 1822, 1942, 2034, 2143, 2261`). `DispatchCtx`
has exactly one definition, `bind/mod.rs:1114`, gated `#[cfg(feature =
"bridge")]`. There is no second definition and no `not(bridge)` arm.

So `cargo test -p driver-cuda --features cuda-12` — `ci.yml:333` — **cannot
compile as the tree stands.** This is stated as a structural claim, not a
build result: one gated definition, 27 ungated uses, one module tier apart.

**And the file says why it happened.** `service.rs:44-49` argues, correctly,
that nothing about this module needs to link:

> *"There is no `#[link]`, no `build.rs` flag and no header … the
> `cargo:rustc-link-lib=cublas` in `build.rs` is for the C++ ARCHIVE's
> remaining callers, not for this file, and it is why lifting it out of the
> `bridge` block changes nothing about what this module needs."*

Every word of that is true **about linking**. The module was moved out of the
`bridge` block on a link argument, and it carried a `bridge`-gated *type* out
with it. The reasoning checked the `-l` and not the `use`.

Its own test knows: `[[test]] gemm_service_parity` has
`required-features = ["bridge"]` (`Cargo.toml:413-414`). **The test that
covers `service.rs` is gated on the feature `service.rs` is not** — which is
exactly the arrangement that lets the break sit unnoticed, because no green
job ever compiles the module without the type.

*Fix shape (not applied — `bind/` is not mine to edit this turn):* `ctx:
&DispatchCtx` is used by these 26 fns for `ctx.stream` and `ctx.cublas`. The
smallest correct change is to take those two `*mut c_void` directly, or a
small ungated `CublasCtx { stream, cublas }`, which makes `service.rs` outlive
`bridge` with no gate at all. That is the shape step 6 wants anyway.

---

## 5. Generated code — the `OUT_DIR` sites

`driver-cuda/build.rs:57-62` is the whole story:

```rust
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_BRIDGE").is_none() { return; }
    bridge::build();
}
```

Everything below is generated only when `bridge` is on, and every `include!`
of it is inside a `bridge`/`jit-parity` gate. The pairing is exact — no
orphans in either direction.

| generated file | written by | `include!`d at | host's gate | step 6 |
|---|---|---|---|---|
| `launch_bindings.rs` | `driver-cuda/build.rs:503-504` (`emit_rust_bindings`) | `bind/abi.rs:329` | `pub mod ffi` — `bridge` (`:318`) | **dies** |
| `rust_dispatch.rs` | `driver-cuda/build.rs:519-520` (`emit_rust_dispatch`) | `bind/mod.rs:2142` | inside `dispatch_generated` — `bridge` (`:1601`) | **dies** |
| `rust_dispatch_probe.rs` | `driver-cuda/build.rs:532-533` (`emit_rust_dispatch_probe`) | `bind/mod.rs:2144` | `jit-parity` (`:2143`) | **dies** |
| `ffi.rs` | `kernels-cuda/build.rs:284-287` (`emit_rust_bindings_portable`) | — (consumed inside `kernels-cuda`) | `native` | **dies** |
| `shim.cpp` | `kernels-cuda/build.rs:270-273` (`emit_c_shim`) | — (compiled by `cc`) | `native` | **dies** |

`api.rs` is already gone (north star §6 half A). `carried.rs`
(`kernels-cuda-new/src/source.rs:227`) and `sources.rs` (`kernels-wgpu`) are
JIT-side and unrelated to `bridge`.

Also dying with `bridge::build()`: the four consistency gates at
`driver-cuda/build.rs:237, 261, 293, 313`, which cross-check
`rust_dispatch.rs` against the shim's defined entry points and against
`JIT_DISPATCHED`. **Those are real coverage.** Their subject
(`pie_k_*` symbols) disappears at the same instant, so they retire with it
rather than needing a home — but they should be deleted *knowingly*, and this
row is the record that they existed.

---

## 6. The shim's link closure

| artefact | where | dies or moves | what notices if it is wrong |
|---|---|---|---|
| `shim.cpp` generation + `cc` compile → `libpie_launch_shim.a` | `kernels-cuda/build.rs:194-304` | **dies** | nothing — `cc` is `native`'s only dependency |
| `println!("cargo:launch_shim={}")` | `kernels-cuda/build.rs:314` | **dies** | `driver-cuda/build.rs:712` `.expect()` — a build-time panic with a written message |
| `links = "pie_kernels_cuda"` | `kernels-cuda/Cargo.toml:21` | **dies** | its own comment `:17` — *"`LAUNCH_SHIM` is the only survivor"*; when that goes, the `links` key has no key left to publish |
| `DEP_PIE_KERNELS_CUDA_LAUNCH_SHIM` read | `driver-cuda/build.rs:712-717` | **dies** | — |
| `cargo:rustc-link-lib=static=pie_launch_shim` + search path | `driver-cuda/build.rs:727-728` | **dies** | — |
| `cargo:rustc-link-lib=static=pie_kernels_cuda` | `driver-cuda/build.rs:752` | **already commented out** | the archive `-l` is gone ahead of the archive |
| the CUDA link closure: `cudart`, `cublas`, `cublasLt`, `cuda`, `nccl`, `stdc++`, `pthread`, `m`, `dl`, `rt` + `lib64` and `stubs` search paths | `driver-cuda/build.rs:775-787` | **dies** | **nothing.** Its own comment names it *"the link closure the shim needs"*. `cudarc` resolves `cudart`/`cublas` by `dlopen` (`fallback-dynamic-loading`), and NCCL appears in Rust only as `nccl_unique_id_hex: String` (`layout/memory_planner.rs:118`, `layout/rendezvous.rs:53`) — a rendezvous identifier, no FFI. **No Rust symbol needs this block.** |
| `rerun-if-changed` on `csrc/src`, `../kernels-cuda-new/csrc/{src,shim}` | `kernels-cuda/build.rs:332,345,346` | **dies with `shim()`** | staleness only |
| `dep:cc` | `kernels-cuda/Cargo.toml` | **dies** | `native` has no other content |

**After this, `kernels-cuda/build.rs` has no `native` half at all**, and
`kernels-cuda`'s `build.rs` + `links` + `[build-dependencies]` reduce to
nothing. That is the crate-level version of the same result `csrc/CMakeLists.txt`
already reached.

---

## 7. The emitters — which lose their last caller

`kernels-cuda-new/src/abi.rs` survives; the question was which *functions*.
Answer: **none of the seven loses every caller**, because
`driver-cuda/tests/launch_abi.rs` and `tests/executor_bind.rs` both require
only `_cuda` (`Cargo.toml:514-515`, `:523`) and both call emitters directly.

| emitter | def | callers that die | callers that survive | verdict |
|---|---|---|---|---|
| `emit_c_shim` | `abi.rs:132` | `kernels-cuda/build.rs:270`; `kernels-cuda/tests/sources.rs:2303` (dies with the crate) | `launch_abi.rs:289,490,1446,2053`; `abi.rs` self-tests `:743,762,782`; `kernels-cuda-new/tests/device_typecheck_types.rs:108` | **survives — test-only** |
| `emit_rust_bindings` | `abi.rs:266` | `driver-cuda/build.rs:503` | `launch_abi.rs:1387`; `abi.rs:744,764`; `device_typecheck_types.rs:120` | **survives — test-only** |
| `emit_rust_bindings_portable` | `abi.rs:287` | `kernels-cuda/build.rs:285` | `device_typecheck_types.rs:127` | **survives — test-only, and its last production consumer is the one that dies** |
| `emit_rust_dispatch` | `abi.rs:1482` | `driver-cuda/build.rs:519` | `executor_bind.rs:1405`; `abi.rs:2165,2196,2379,2530`; `device_typecheck_types.rs:133,159` | **survives — test-only** |
| `emit_rust_dispatch_probe` | `abi.rs:1520` | `driver-cuda/build.rs:532` | `kernels-cuda-new/examples/dispatch_countdown.rs:466` | **survives on an example alone** — the weakest consumer set of the seven |
| `emit_device_typecheck` | `abi.rs:383` | — | `kernels-cuda/examples/emit_device_typecheck.rs:81`; `device_typecheck_types.rs:184,210,359,536,558`; `abi.rs:2592,2619,2651,2677` | **unaffected by `bridge`** |
| `emit_layout_assertions` | `abi.rs:670` | — | `launch_abi.rs:1880,1902,1963,1986` | **survives — the production reason `abi.rs` stays** |

Two things this table says that prose would bury:

1. **Four of the seven emitters keep only test callers after step 6.** They do
   not become dead code — `launch_abi.rs` is the tree's remaining compiler of
   the mirrored headers and it needs `emit_c_shim` to build the text it
   compiles — but their *production* consumer set becomes empty, and that is a
   different status from "alive". If `launch_abi.rs` is ever narrowed, check
   this column first.
2. **`emit_device_typecheck` is the emitter `bridge` never touched**, and its
   consumer set is entirely offline. Its non-test caller,
   `kernels-cuda/examples/emit_device_typecheck.rs`, **does not spawn nvcc** —
   it writes text and prints the `nvcc -std=c++20 -arch=sm_89 -fatbin …` line
   for a human to run (`:12`). Its test caller
   `kernels-cuda-new/tests/device_typecheck_types.rs` **does** spawn one.

   *Correction to a claim this sweep first made and then checked:* that test
   is not the last nvcc invoker in the tree. Six files spawn nvcc, all in
   `kernels-cuda-new` and all offline probes: `examples/{fp8_pipeline_probe,
   halftype_parity, mma_probe}.rs` and `tests/{device_typecheck_types,
   flashinfer_decode, plan}.rs`. None is on a build path and none compiles a
   shipped translation unit, so nvcc-zero stands — but "no `.cu` in the tree"
   and "nothing runs nvcc" are two different statements and only the first is
   true. The distinction matters for anyone grepping `nvcc` to confirm the
   milestone.

---

## 8. Recorded, not fixed

Per instruction, findings in other agents' files are recorded here rather than
edited.

| finding | where | status |
|---|---|---|
| `bind/service.rs` depends on the `bridge`-gated `DispatchCtx` from a `_cuda`-only module — 27 sites | `bind/service.rs:72` + 26 signatures | **live break**, §4 |
| `[[test]] gemm_service_parity` requires `bridge` while its subject module does not | `driver-cuda/Cargo.toml:413-414` | the reason §4 is invisible in CI |
| `service.rs:44-49` justifies leaving the `bridge` block on a *linking* argument that is true, and misses the *type* dependency | `bind/service.rs:44-49` | root cause of §4 |
| `kernels-cuda/tests/sources.rs:213` asserts `csrc/src` non-empty with the message *"…`kernels.def` live here and are still read"* — second half false since the manifest reader retired | `kernels-cuda` — `fimoe-rust`'s | carried from the previous turn, still true |
| `kernels/src/lib.rs:1078` — `Ty::YarnOriginalParams` spells a C++ path for a struct with no declaration | inert only because the row is `RUST_SERVED` | carried, still true |

---

## Method, so this can be disagreed with

1. Gate sites: every `#[cfg(…)]`/`#![cfg(…)]` whose predicate names `bridge`
   or `jit-parity`, in `driver-cuda/src` and `kernels-cuda`.
   `driver-wgpu`/`driver-vulkan` were **excluded**: they carry features of the
   same names that are unrelated to this one.
2. Item spans: from the attribute to the close of the item, by brace count
   over string- and comment-stripped lines. A first pass that assumed the
   body opened on the signature line **undercounted spans and inflated the
   ungated-reference count from 69 to 645** — nested helpers inside
   `dispatch_generated` (`:1601-2146`) and `dispatch` (`:2225-2717`) read as
   top-level. If a number here looks large, that is the failure mode to check.
3. Reference matching: word-boundary for type names; call- or path-shaped
   patterns for function names, because bare `run`, `dispatch`, `apply`,
   `stage`, `plan` and `ffi` collide with `program::run`, `core::ffi`, local
   `let (mut dispatch, …)` bindings and `XqaMember::dispatch`. Every
   lowercase name in §2 was disambiguated individually.
4. Tier assignment is by **module declaration**, not by file contents — a file
   is tier B because `fire/mod.rs` or `lib.rs` gates its `mod` line, whatever
   is written inside it.
5. **Nothing was compiled.** Where this document says a configuration does not
   build (§4), that is an argument from one gated definition and 27 ungated
   uses, and it is offered as a claim to check, not a result.
