---
title: "Forward-pass WIT refactor — per-architecture pass interfaces"
---

# Forward-pass WIT refactor

Status: **design ratified, not implemented.**
Scope: `interface/inferlet/` WIT surface, `sdk/{rust,python,javascript}`,
`runtime/engine/src/inferlet/host/`, PTIR container identity.

---

## 1. Motivation

### 1.1 The gap this closes

Pie gives first-class programmability to attention KV state and almost none to
recurrent state (RS). The RS path exists end to end — `rs-working-set` in
`working-set.wit`, `RsStore` in `runtime/engine/src/store/rs.rs`, the
`rs_fold_lens` / `rs_buffer_slot_ids` / `rs_buffer_slot_indptr` wire fields in
`interface/driver/`, and the GDN `commit_len` fold path in
`driver/cuda/src/model/qwen3_5/` — but the *guest-facing* surface treats it as
an afterthought bolted onto an attention-shaped API.

Concretely, of the six capabilities `kv-working-set` exposes, `rs-working-set`
has none:

| KV capability | RS counterpart |
|---|---|
| `discard(ranges)` — arbitrary-range eviction | none (`free-buffer` only drops *unfolded* slots) |
| `slice(range)` — partial rebase | none |
| `copy-into(...)` — token-cell moves | none (`reorder-buffer` is page-granular) |
| `update-index` / `from-index` / `remove-index` — global prefix CAS | none (explicitly out of v1) |
| host swap / offload | none (`store/rs.rs` has zero `swap` sites vs. 26 in `store/kv.rs`) |
| `attn_score` intrinsic observation | none — no recurrent tap exists |

The API shape is a first-order cause of several of these. This refactor fixes
the shape; the storage-side gaps (RS prefix CAS, RS swap) are tracked
separately.

### 1.2 Concrete defects in the current surface

Evidence from `runtime/engine/tests/inferlets/gdn-foldcommit/src/lib.rs`:

1. **Dummy geometry is mandatory.** `forward.wit` documents `fold-buffered` as
   *"Runs ONLY the recurrent layers (no embed, no attention/MLP, no logits)"*,
   yet the commit fire must still bind `embed(...)`, a full 10-argument
   `attention(ws, .., kv_len, pages, page_indptr, w_slot, w_off, positions,
   None)`, and an epilogue materializing `logits` (lines 160–212). The contract
   and the surface disagree.

2. **Legality is a runtime error, not a type.** `set-rs-mode` is rejected on a
   pure-attention model. `on_attn` is meaningless on a recurrent layer. Neither
   is expressible.

3. **Silent misapplication.** `quest-attention`, `trackb-h2o`, `trackb-snapkv`,
   and `tova-attention` are attention-only algorithms. Running them on a
   GDN/hybrid model is *semantically wrong* — evicting KV pages does not undo
   the folded recurrent state's absorption of those tokens — and nothing
   catches it today.

4. **No recurrent tap.** PTIR stages are `Prologue` / `OnAttnProj` / `OnAttn` /
   `Epilogue` (`compiler/ir/src/registry.rs:15-24`); the per-layer taps are
   attention-only. Adding `OnRecurrent` today means bumping `PTIR_VERSION` and
   the frozen wire tags in `ptir_abi.h` — a change that hits every guest,
   including pure-attention ones.

5. **The SDK pre-flights port binding by hand.** `attach_program()`
   (`sdk/rust/inferlet/src/ptir.rs:944-960`) walks a hardcoded `required[]` list
   and re-derives "descriptor channels are missing" before the host ever sees
   the pass, duplicating a check the host already performs in
   `validate_descriptor_bindings` (`runtime/engine/src/inferlet/host/forward.rs`)
   with strictly more information.

   **Note — `TraceContainer.ports` is NOT redundant with the WIT calls, and
   this refactor does not remove it.** The two carry different information and
   are cross-checked against each other:
   - `PortSource::Const` (`container.rs:106-116`) folds a trace-known
     rectangular `indptr` / `positions` / `readout` into the container. WIT
     accepts only `borrow<channel>`, so a const port has **no WIT expression at
     all**.
   - The *host* writes ports: `pipeline/fire/{geometry,kv,shadow}.rs` and
     `host/forward.rs` synthesise `PortSource::Const` entries during fire
     preparation, and `compiler/codegen/src/launch.rs:197` (`lower_ports`)
     lowers them to `LaunchPort { is_const, .. }` — which is what reaches the
     C++ driver as `PieLaunchPort` (`driver/abi/include/pie_native/launch/program.hpp:310`).
   - `compiler/eval` (`pareval.rs`, `interp/mod.rs`) reads `container.ports` as
     the reference evaluator's input binding.

   So `ports` is the canonical lowering form that the WIT calls *feed*, not a
   duplicate of them. Only the guest-side `required[]` pre-check is deleted.

---

## 2. Design decisions

### D1 — Split the forward pass by **state semantics**, one WIT interface each

Three interfaces: `forward` (paged KV only), `forward-recurrent` (folded
recurrent state only), `forward-hybrid` (both).

Rejected alternative: a single `forward-pass` with a
`variant state-binding { paged-kv | latent-kv | recurrent }`. Adding a case to
a WIT variant is a **breaking** change; adding a method to a resource is
**additive**. Splitting into resources confines the blast radius of every
future change to one interface.

The taxonomy is over *state semantics*, not model families. `attention /
recurrent / hybrid` is closed and slow-moving; `architecture()` is an open,
fast-growing string set (13 families in `driver/cuda/src/model/registry.cpp`).
MLA's radically different per-page byte layout (`mla_paged.cu`,
`mla_cache_view.hpp`) was absorbed with **zero** WIT change, which is the
empirical proof that the axis is right: the engine's KV store knows only
`kv_page_size` (`runtime/model/src/lib.rs`).

### D2 — Keep the resource name `forward-pass` in all three interfaces

WIT scopes resource names to their interface, so three interfaces may each
declare `forward-pass`. Generated bindings differ only by module path
(`wit::forward_recurrent::ForwardPass`, `imports/forward_recurrent.py`,
`interfaces/pie-core-forward-recurrent.d.ts`). All three SDKs already use
per-interface module layouts.

This collapses the migration cost: `ForwardPass::new` has **133 call sites
across 68 files**, and all of them keep compiling behind a single changed
`use` line.

### D3 — No shared `pass-body`; duplicate declarations across interfaces

`embed`, `readout`, `prologue`, `epilogue` are declared three times. This is
deliberate.

WIT duplication does not force implementation duplication: the host keeps one
private `PassCore` in `runtime/engine/src/inferlet/host/` shared by all three
resource impls. What is duplicated is ~24 lines of declaration plus generated
bindings.

What it buys, beyond independent evolution: a `constructor(body: pass-body)`
would create an ownership puzzle (borrowed or owned? can two passes consume one
body? can hooks be attached to the body after a pass consumes it?), all of
which would need runtime guards. Full separation makes those states
unrepresentable.

### D4 — Hooks become pass methods; stage bodies stay as bytes

The user-facing SDK already exposes hooks as pass methods
(`ptir.rs:894-907`). The collapse into one opaque blob happens at
`attach_program()`, which encodes every stage into a single `TraceContainer`
and ships it through `program(container_bytes, channels)`.

Of the three reasons for that blob, only the first is essential:

- **(a) WIT cannot carry closures** — stated in the current `forward.wit`
  header. A hook is a traced closure; the wire artifact must already be
  serialized code. **Irreducible.**
- **(b) Identity is an FNV-1a hash over canonical container bytes**
  (`container.rs:6-9`, contract C3). If stages arrived as separate calls the
  host would have to reassemble and canonicalize, i.e. own the encoder.
  **Implementation choice, not essential.**
- **(c) Cross-stage validation is whole-pass** — the global per-channel program
  order is *stage order, then op order within a stage*, with the descriptor
  phase interleaved (`registry.rs:34-38`, `Phase::ORDER`). **Real, but it can
  run at first submit instead of at `program()`.**

So: **stage topology becomes first-class on the pass; stage bodies stay as
bytes, one container per stage.** `TraceContainer.stages` is already
`Vec<StageProgram>` documented as *"Sorted by stage tag, unique (at most one
program per stage)"* — a stage-keyed map. Splitting it is repackaging, not
redesign.

Consequences:
- `on-recurrent` is added to `forward-recurrent` / `forward-hybrid` as a plain
  additive method. `forward`'s ABI never moves.
- Pass identity becomes the hash over sorted `(stage-tag, stage-hash)` pairs.
  This enables **per-stage compile caching**: a sampler epilogue reused across
  different attention taps now hits cache; today the single pass hash misses.
- Port/channel *declaration* stays in the container (see §1.2 note 5 — it is
  the lowering form, and const ports have no WIT expression). What is deleted
  is the SDK's hand-rolled `required[]` pre-check in `attach_program()`; the
  host's `validate_descriptor_bindings` remains the single enforcement point.

### D5 — `rs-mode` variant becomes three methods

`fold()` / `buffer()` / `fold-buffered(lens)`. Same reason as D1: adding a fold
strategy later is additive on a resource, breaking on a variant. This removes
the highest-evolution-risk type in the current `forward.wit`.

### D6 — The state-binding method is named `attention` in all three interfaces

Uniform slot name; only the signature varies. The role inside a pass is the
same regardless of whether the mechanism is softmax attention or a recurrence.

### D7 — Hybrid binds KV and RS in **one** `attention` call

They are one set of geometry describing different layers of the same forward,
not two independent decisions.

The KV half is `option<kv-binding>`. The one case where it is `none` is a
`fold-buffered` fire, which runs recurrent layers only. The RS half stays
required even there — the driver needs the folded slot plus the buffered CSR.

This optional is what preserves **frame monomorphism**: `submit`'s
`slots: list<option<borrow<forward-pass>>>` is monomorphic, and WIT has no
existential types, so a hybrid frame mixing a recurrent-only fire with a normal
fire is only expressible if both are the same resource type.

### D8 — Geometry folded into records

`kv-geometry` (9 fields) and `rs-geometry` (7 fields) replace flat parameter
lists. A fully flattened hybrid `attention` would take 18 positional
parameters, and the KV-half optionality would have no grouping to attach to.

The records are **declared separately per interface** rather than shared, for
the same reason as D3.

**Toolchain-verified.** `borrow<T>` inside a record — including the nested
`kv-binding { borrow<kv-working-set>, kv-geometry }` under an `option`, plus
`list<borrow<rs-working-set>>` — was confirmed to parse and generate on both
sides of the boundary at the versions this repo pins:

- guest, `wit-bindgen 0.59.0` → `struct KvBinding<'a> { working_set: &'a KvWorkingSet, geometry: KvGeometry<'a> }`
  and `fn attention(&self, kv: Option<&KvBinding<'_>>, rs: &[&RsWorkingSet], rs_geom: &RsGeometry<'_>)`;
- host, `wasmtime 47.0.1` → `struct KvGeometry { kv_len: Resource<Channel>, mask: Option<Resource<Channel>> }`,
  i.e. the same `Resource<T>` the existing `borrow<channel>` parameters already
  produce.

Handles flatten to scalar indices in the canonical ABI, so the records cost
nothing on the wire relative to the flat parameter lists they replace.

### D9 — RS gets geometry, mirroring KV

The buffered RS slots are now addressed explicitly by channels
(`buffer-pages` / `buffer-indptr` as CSR, `w-slot` / `w-off` for the write
side), mapping 1:1 onto the existing `rs_buffer_slot_ids` /
`rs_buffer_slot_indptr` wire fields. Previously the buffer write position was
an implicit `buffer(start-token: u32)` with a page-alignment precondition and a
"driver fills page-major from that offset" convention.

### D10 — `model.pass-kind()` selects the interface

A closed enum, not a parse of `architecture()`. Constructing the wrong
interface's `forward-pass` errors immediately.

### D11 — No deprecation window

`forward` keeps its name, so its `attention()` signature change breaks
pure-attention guests too. **Accepted.** Name continuity and a deprecation
window are mutually exclusive; we take the name. There is no `forward-legacy`.

---

## 3. Final WIT

Files under `interface/inferlet/`:

```
channel.wit            new — shared
forward.wit            name kept, contents replaced (attention-only)
forward-recurrent.wit  new
forward-hybrid.wit     new
model.wit              + pass-kind
working-set.wit        unchanged
world.wit              imports updated
```

### 3.1 `channel.wit`

```wit
// All three forward-* interfaces reference this, so it must be a single
// definition — declaring it per interface would make three incompatible
// types and no channel handle could be passed to more than one.
interface channel {
    use types.{error, shape, dtype, data};

    resource channel {
        constructor(shape: shape, dtype: dtype, capacity: u32);
        put:  func(value: data) -> result<_, error>;
        set:  func(value: data) -> result<_, error>;
        take: async func() -> result<data, error>;
        read: async func() -> result<data, error>;
    }
}
```

### 3.2 `forward.wit` — attention-only

```wit
// llama_like · gemma* · mixtral · kimi · deepseek_v4 · glm5 · qwen3_vl · csm.
// Valid only when `model.pass-kind()` == attention.
//
// The resource name `forward-pass` is shared by all three interfaces: WIT
// scopes resource names to their interface, so this is legal and the
// generated bindings differ only by module path.
interface forward {
    use types.{error, data};
    use channel.{channel};
    use working-set.{kv-working-set, page-span};
    use pipeline.{pipeline};

    /// Attention geometry. forward-hybrid declares its own copy on purpose:
    /// adding a field on one side must not move the other's ABI.
    record kv-geometry {
        readable-pages: page-span,
        writable-pages: page-span,
        kv-len:      borrow<channel>,
        pages:       borrow<channel>,
        page-indptr: borrow<channel>,
        w-slot:      borrow<channel>,
        w-off:       borrow<channel>,
        positions:   borrow<channel>,
        mask:        option<borrow<channel>>,
    }

    resource forward-pass {
        constructor();

        /// State binding. Required.
        attention: func(kv: borrow<kv-working-set>, geom: kv-geometry)
            -> result<_, error>;

        embed:   func(tokens: borrow<channel>, indptr: borrow<channel>)
            -> result<_, error>;
        readout: func(indices: borrow<channel>) -> result<_, error>;

        // ── Hooks (stages) ───────────────────────────────────────────
        // Each argument is the canonical PTIR bytes of exactly ONE stage
        // plus its channel handles in dense declaration order. WIT cannot
        // carry closures, so the body stays bytes — but WHICH hooks are
        // legal is now stated by the type.
        //
        // Pass identity = hash over the sorted (stage-tag, stage-hash) pairs.
        prologue:     func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
        on-attn-proj: func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
        on-attn:      func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
        epilogue:     func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
    }

    /// Submit ONE FRAME: exactly `model.frame-size()` ordered slots. Slot i
    /// executes in wave i; `none` is a no-op for that wave. Validation is
    /// deterministic and structural, never timing-dependent.
    submit: func(
        on: borrow<pipeline>,
        slots: list<option<borrow<forward-pass>>>,
    ) -> result<_, error>;
}
```

### 3.3 `forward-recurrent.wit` — linear/SSM only

```wit
// Recurrent layers only (no attention layers).
// Valid only when `model.pass-kind()` == recurrent.
interface forward-recurrent {
    use types.{error, data};
    use channel.{channel};
    use working-set.{rs-working-set, page-span};
    use pipeline.{pipeline};

    /// The counterpart of `kv-geometry`. Addresses the RS working set's
    /// BUFFERED slots. Every channel value is a WorkingSet-relative buffer
    /// page index; the runtime translates them to physical slots and lowers
    /// them as `rs_buffer_slot_ids` / `rs_buffer_slot_indptr`. The folded
    /// state slot comes from the working-set handle and is not named here.
    record rs-geometry {
        readable-buffer: page-span,       // range `fold` may read
        writable-buffer: page-span,       // range `buffer` may write
        buffer-len:    borrow<channel>,   // live buffered tokens per request
        buffer-pages:  borrow<channel>,   // CSR values — buffer page ids
        buffer-indptr: borrow<channel>,   // CSR row bounds, one per request
        w-slot:        borrow<channel>,   // buffer page each token writes to
        w-off:         borrow<channel>,   // offset within that page
    }

    resource forward-pass {
        constructor();

        /// State binding. Required. The mechanism is a recurrence, but the
        /// SLOT NAME is `attention` in all three interfaces — only the
        /// signature varies.
        attention: func(
            rs:   list<borrow<rs-working-set>>,   // resolved request order
            geom: rs-geometry,
        ) -> result<_, error>;

        // ── Fold mode ────────────────────────────────────────────────
        // The former `rs-mode` variant, flattened into methods: adding a
        // variant case is breaking, adding a resource method is additive.
        // Exactly one of the three (defaults to `fold` if none is called).

        /// Fold every token of this fire into the folded state, in-forward
        /// and IRREVERSIBLY. The plain prefill/decode path and the default.
        fold: func() -> result<_, error>;

        /// Write each token's pre-recurrence activations into the buffered
        /// slots named by `geom.w-slot` / `geom.w-off`, leaving the folded
        /// state UNTOUCHED. This is what makes a linear model speculatable:
        /// an uncertain tail that is never folded costs nothing to abandon.
        /// Requires an already-materialized folded state.
        buffer: func() -> result<_, error>;

        /// Replay buffered tokens into the folded state: one length per
        /// bound working set, in request order. Runs ONLY the recurrent
        /// layers, advances the folded boundary, and drops fully covered
        /// head slots. Each length must be a positive multiple of
        /// `model.rs-fold-granularity()`.
        fold-buffered: func(lens: list<u32>) -> result<_, error>;

        // Both may be omitted under `fold-buffered` (embed does not run).
        embed:   func(tokens: borrow<channel>, indptr: borrow<channel>)
            -> result<_, error>;
        readout: func(indices: borrow<channel>) -> result<_, error>;

        // ── Hooks (stages) ───────────────────────────────────────────
        prologue: func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;

        /// Per-recurrent-layer tap — the counterpart of `on-attn-proj` /
        /// `on-attn`. It exists only in this interface, so adding it never
        /// moves `forward`'s ABI.
        on-recurrent: func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;

        /// Required under `fold` / `buffer`. OPTIONAL under `fold-buffered`,
        /// and under that mode materializing `logits` / `mtp-logits` /
        /// `hidden` / `value-head` is rejected at submit — that fire does
        /// not compute logits.
        epilogue: func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
    }

    submit: func(
        on: borrow<pipeline>,
        slots: list<option<borrow<forward-pass>>>,
    ) -> result<_, error>;
}
```

### 3.4 `forward-hybrid.wit` — attention + recurrent layers in one forward

```wit
// Attention layers and recurrent layers coexist in one forward
// (Qwen3.5 GDN dense/MoE, Nemotron-H Mamba2).
// Valid only when `model.pass-kind()` == hybrid.
interface forward-hybrid {
    use types.{error, data};
    use channel.{channel};
    use working-set.{kv-working-set, rs-working-set, page-span};
    use pipeline.{pipeline};

    /// Deliberately DISTINCT types from `forward.kv-geometry` and
    /// `forward-recurrent.rs-geometry` — adding a field on one side must
    /// not move the other's ABI.
    record kv-geometry {
        readable-pages: page-span,
        writable-pages: page-span,
        kv-len:      borrow<channel>,
        pages:       borrow<channel>,
        page-indptr: borrow<channel>,
        w-slot:      borrow<channel>,
        w-off:       borrow<channel>,
        positions:   borrow<channel>,
        mask:        option<borrow<channel>>,
    }

    record rs-geometry {
        readable-buffer: page-span,
        writable-buffer: page-span,
        buffer-len:    borrow<channel>,
        buffer-pages:  borrow<channel>,
        buffer-indptr: borrow<channel>,
        w-slot:        borrow<channel>,
        w-off:         borrow<channel>,
    }

    /// Groups the KV half so it can be made optional as a unit. Without this
    /// wrapper a `fold-buffered` fire could not omit KV without inventing
    /// dummy geometry.
    record kv-binding {
        working-set: borrow<kv-working-set>,
        geometry:    kv-geometry,
    }

    resource forward-pass {
        constructor();

        /// State binding — a hybrid binds KV and RS in ONE call. They are
        /// one set of geometry describing different layers of the same
        /// forward, not two independent decisions.
        ///
        /// `kv` is `none` in exactly one case: a `fold-buffered` fire, which
        /// runs recurrent layers only and therefore has no KV geometry. The
        /// RS half stays required even there — the driver needs the folded
        /// slot plus the buffered CSR.
        ///
        /// This optional is what preserves FRAME MONOMORPHISM: every slot of
        /// a hybrid frame is the same `forward-pass` type, so a
        /// recurrent-only fire and a normal fire can share a frame. (WIT has
        /// no existential types, so a heterogeneous slot list is
        /// inexpressible.)
        attention: func(
            kv:      option<kv-binding>,
            rs:      list<borrow<rs-working-set>>,
            rs-geom: rs-geometry,
        ) -> result<_, error>;

        // ── Fold mode (same contract as forward-recurrent, redeclared) ──
        fold:          func() -> result<_, error>;
        buffer:        func() -> result<_, error>;
        fold-buffered: func(lens: list<u32>) -> result<_, error>;

        embed:   func(tokens: borrow<channel>, indptr: borrow<channel>)
            -> result<_, error>;
        readout: func(indices: borrow<channel>) -> result<_, error>;

        // ── Hooks (stages) — all five ────────────────────────────────
        prologue:     func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
        /// Fires on attention layers only.
        on-attn-proj: func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
        /// Fires on attention layers only.
        on-attn:      func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
        /// Fires on recurrent layers only.
        on-recurrent: func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
        /// Optional under `fold-buffered`; logits-class intrinsics rejected
        /// under that mode.
        epilogue:     func(program: data, channels: list<borrow<channel>>)
            -> result<_, error>;
    }

    submit: func(
        on: borrow<pipeline>,
        slots: list<option<borrow<forward-pass>>>,
    ) -> result<_, error>;
}
```

### 3.5 `model.wit` — addition

```wit
    /// Which forward-pass kind the bound model requires. A guest may only
    /// construct the `forward-pass` of the matching interface; the other
    /// interfaces' constructors error immediately.
    ///
    /// Do NOT derive this by parsing `architecture()` — that is an open set
    /// and every new model family would break guests. This enum is closed
    /// over state semantics, so it does not grow with the model zoo.
    enum pass-kind {
        attention,   // per-token, reversibly discardable KV  -> forward
        recurrent,   // irreversibly folded state             -> forward-recurrent
        hybrid,      // both                                  -> forward-hybrid
    }
    pass-kind: func() -> pass-kind;
```

`is-linear()` remains, but only as a commit-policy hint (equivalent to
`pass-kind() != attention`). All binding selection goes through `pass-kind()`.

### 3.6 `world.wit`

```wit
package pie:inferlet@0.3.0;

world inferlet {
    import types;
    import model;
    import tokenizer;
    import pipeline;
    import working-set;

    import channel;
    import forward;             // attention-only
    import forward-recurrent;
    import forward-hybrid;

    import grammar;
    import chat;
    import tools;
    import reasoning;
    import media;
    import speech;
    import session;
    import system;

    // wasi 0.3 surfaces — unchanged

    export run;
}
```

---

## 4. Summary table

| | `forward` | `forward-recurrent` | `forward-hybrid` |
|---|---|---|---|
| State binding | `attention(kv, kv-geometry)` | `attention(rs, rs-geometry)` | `attention(option<kv-binding>, rs, rs-geometry)` |
| Records | `kv-geometry` | `rs-geometry` | `kv-geometry`, `rs-geometry`, `kv-binding` |
| Fold mode | — | `fold` / `buffer` / `fold-buffered` | same |
| Hooks | prologue, on-attn-proj, on-attn, epilogue | prologue, **on-recurrent**, epilogue | all five |
| Submit | interface-level free func over its own slot type | same | same |

---

## 5. SDK shape

```rust
// sdk/rust/inferlet/src/ptir.rs
pub mod attention { pub struct ForwardPass { .. }  pub mod prelude { .. } }
pub mod recurrent { pub struct ForwardPass { .. }  pub mod prelude { .. } }
pub mod hybrid    { pub struct ForwardPass { .. }  pub mod prelude { .. } }
```

**No unprefixed `ptir::prelude`.** Keeping one would let a wrong pass be
selected silently, which defeats the correctness gate this refactor exists to
build. Leave the old path as a module that fails to compile, not as an alias to
`hybrid`.

`.submit()` stays as SDK sugar (`submit_frame(on, &[Some(self)])`); the WIT
`submit` remains an interface-level free function so callers never write
`a.submit(pipe, [a, b, c])`.

Per-inferlet migration is one line:

```rust
-use inferlet::ptir::prelude::*;
+use inferlet::ptir::attention::prelude::*;
 ...
 let fwd = ForwardPass::new();     // unchanged, x126
```

Python and JavaScript bindings follow their existing per-interface layouts
(`imports/forward_recurrent.py`, `interfaces/pie-core-forward-recurrent.d.ts`).

---

## 6. What this fixes

| Defect (§1.2) | Resolution |
|---|---|
| `rs-mode` variant is evolution-hostile | Deleted; three additive methods |
| 10 positional args on `attention` | `kv-geometry` record |
| Dummy attention geometry on `fold-buffered` | `kv` is `none` on hybrid; omitted entirely on recurrent |
| Hook legality is a runtime error | Typed — `on-attn` exists only where attention layers do |
| Adding `on-recurrent` needs a `PTIR_VERSION` bump | Additive method; `forward`'s ABI is unmoved |
| H2O / Quest / TOVA silently wrong on GDN | Blocked at the `attention::prelude` import |
| SDK hand-rolls a `required[]` port pre-check | Deleted; the host's `validate_descriptor_bindings` is the only enforcement point (`TraceContainer.ports` is retained — §1.2 note 5) |
| One pass hash ⇒ compile-cache miss on any change | Sorted `(stage, hash)` pairs ⇒ per-stage caching **(deferred, §7 step D)** |

---

## 7. Implementation order

Five landable units. **A** is additive and lands alone; **B** is one atomic
commit; **D** is a separate PR that this refactor does not depend on.

### A — `channel.wit` extraction + `model.pass-kind()`  *(standalone, additive)*

Move the `channel` resource out of `forward.wit` into `interface/inferlet/channel.wit`
and add the `pass-kind` enum + func to `model.wit`. `channel` is referenced
nowhere outside `forward.wit` today, so the extraction introduces no interface
cycle. Landing A first materially shrinks B's diff.

### B — the three interfaces  *(one atomic commit)*

`forward.wit` keeps its name but changes `attention`'s signature, which breaks
the SDK, which breaks every inferlet. There is no intermediate state where the
old and new surfaces coexist (D11), so **WIT + host + Rust SDK + inferlets must
land together**:

1. Replace `forward.wit` (attention-only, `kv-geometry`); add
   `forward-recurrent.wit` and `forward-hybrid.wit`; bump the package to
   `0.3.0`; update `world.wit`.
2. Three `forward-pass` resource impls over one private `PassCore` in
   `runtime/engine/src/inferlet/host/`, with a `pass-kind` gate in each
   constructor. Update the `bindgen!` `with:` map and `add_to_linker`.
3. Rust SDK module split (`ptir::{attention,recurrent,hybrid}`). This removes
   `set_rs_working_sets` and `set_rs_mode` (`ptir.rs:847-870`), whose roles are
   absorbed by `attention(..)` and by `fold` / `buffer` / `fold-buffered`.
4. Migrate the inferlets: **133 `ForwardPass::new` call sites across 68 files**,
   one `use` line each.

**Acceptance criterion for the whole refactor:** `trackb-h2o`,
`trackb-snapkv`, `tova-attention`, and `quest-attention` must now fail to
build or bind against a hybrid model. If they still succeed, the split did not
buy what it was for.

### C — delete the SDK `required[]` pre-check  *(landed inside B)*

Small follow-up to B. `TraceContainer.ports` is untouched (§1.2 note 5). In
the event this fell out of B for free: the per-kind pass types made the
hardcoded guest-side port list unreachable, so it was deleted in the same
commit.

### E — Python / JavaScript SDKs

Follow the existing per-interface layouts.

### D — per-stage PTIR containers  *(separate PR, not a dependency)*

Split `TraceContainer.stages` into one container per stage and change pass
identity to the hash over sorted `(stage-tag, stage-hash)` pairs, enabling
per-stage compile caching.

Deliberately **decoupled** from A/B/C/E, because it is the only unit with wire
risk and its cost/benefit is different from the doc's original framing:

- *Cheaper than assumed on the driver side.* There is **no C++ container
  reader** (`grep -rl TraceContainer driver/` is empty). The driver consumes
  the *lowered* `PieLaunchPort` / op-table form via `ptir_abi.h`, not the
  container. (`PTIR-CONTAINER.md`, cited by `ptir_abi.h:3`, is not present in
  the tree — that reference is stale and should be fixed or dropped.)
- *Less valuable than assumed on the guest side.* The SDK already accumulates
  every stage and emits a single `program()` call lazily at first submit
  (`ptir.rs:940`), so per-stage `program()` calls change nothing observable for
  a guest today. The win is purely compile-cache hit rate.
- *Genuinely whole-pass validation.* SPSC direction (T2) and sink stage
  precedence (T11) in `compiler/ir/src/validate.rs` are cross-stage, so they
  must move to first `submit()`.
- Changing the identity hash invalidates every key in the program LRU
  (`runtime/engine/src/pipeline/program.rs:203`).

`on-recurrent` does **not** wait on D: it is a new stage tag plus an additive
WIT method, and lands with B.

---

## 8. Open items

### 8.1 `buffer()` vs `buffer(start-token: u32)` — ~~decide during implementation~~ RESOLVED

**Resolved as `buffer(start-token, rs-geometry)`.** See §9.2 for the argument
and §9.3 for where the geometry ended up. The rest of this section is the
original framing, kept for the record.

D9 gives `rs-geometry` explicit `w-slot` / `w-off` channels, which makes the
former `start-token` argument redundant and removes both the
`rs-buffer-page-size` alignment precondition and the "driver fills page-major
from that offset" convention. The document above assumes the no-argument form.

The cost is on the driver write side: `rs_buffer_slot_ids` population becomes
channel-driven rather than derived from an offset, which touches the plan in
`.wiki/designs/workingset-rs-design.md` §7.4. **Defer the final call to
implementation.**

### 8.2 Not fixed by this refactor

These are shape-independent and tracked separately:

- **`nemotron_h` has no fold path.** `driver/cuda/src/model/nemotron_h/` has
  zero `commit_len` / `rs_fold` / `rs_buffer` sites, yet the model reports as
  linear. Under the new surface `fold-buffered()` is a typed method, so the
  driver at least gains an honest place to reject it via capability.
- **Metal collapses the RS flag bits.** `driver/metal/src/batch/batch_schedule.hpp:114`
  does `rs_slot_flags[r] != 0` and treats the result as RESET, so
  `RS_FLAG_FOLD` would be misread. Metal marshals `rs_fold_lens`
  (`context.cpp:709`) but implements no fold.
- **No RS prefix cache.** `kv-working-set` has `update-index` / `from-index` /
  `remove-index`; `rs-working-set` has no content-addressed sharing, so linear
  and hybrid models get no cross-request prefix reuse.
- **No RS host swap.** `store/rs.rs` has no offload path, so RS slot pressure
  directly caps concurrency and eviction requires lineage replay.
- **No recurrent observation intrinsic.** There is no counterpart to
  `AttnScore` for gates, decay, or conv state, so observation-driven eviction
  and sparsity algorithms have no linear-model analogue. `on-recurrent` creates
  the stage; the intrinsic is a separate addition.

---

## 9. Implementation log — decisions taken while landing Step B

These are the points where implementation contradicted or refined the design
above. They are recorded here because the design is the source of truth and had
to move.

### 9.1 D4 splits into (a) and (b), and only (b) is deferred

The review statement "`on-recurrent` does not wait on D" was imprecise. D4 has
two coupled halves:

- **(a)** hooks as per-stage WIT methods on the pass, and
- **(b)** per-stage containers plus a per-stage identity hash.

(a) without (b) forces the host to reassemble and canonicalize a container from
separately-submitted stages — exactly what D4 rejects as "the host owning the
encoder." They ship together or not at all.

**Step B therefore keeps `program(container-bytes, channels)`.** Hook legality
is still enforced: the host validates the container's stage set against the
interface it was submitted through, so `on_attn` on a `forward-recurrent` pass
is rejected. Methodization moves wholesale to Step D.

### 9.2 `buffer` keeps its `start-token` argument

D5 flattens `rs-mode` into `fold` / `buffer` / `fold-buffered`. It showed
`buffer: func()`, on the assumption that `rs-geometry.w-slot` / `w-off` carry
the write descriptor.

They will — but only once the engine half of the device-resolved RS path
(§9.3) is wired end to end. Today the ONLY working lowering is the host-derived
one, which derives the buffered CSR from `(start-token, row-tokens)` in
`RsStore`. So the shipped signature is `buffer: func(start-token: u32)`.

This is also the strictly safer direction: dropping a parameter later is
breaking either way, but a parameter that becomes redundant can simply be
ignored, whereas ADDING one later is unambiguously breaking.

### 9.3 `rs-geometry` moves off `attention` and onto `buffer` / `fold-buffered`

> **Superseded by §11.3.** The premise below — that the buffer is touched by
> two calls out of many — stops holding once the buffer is understood as half
> the state. The rule ("geometry goes where it is read") survives; the answer
> to *where it is read* moves back to the state binding.

Two changes to D9.

**It is not part of the state binding.** D9 hung `rs-geometry` off `attention`
alongside the working sets. That forces a plain `fold` prefill/decode — which
never touches the buffer — to supply one, which is exactly the
mandatory-dummy-geometry wart of §1.2 defect 1. Making it an `option` there
avoids the wart but reintroduces an unrepresentable-state hole in the other
direction: `some(geometry)` under `fold` is meaningless, and `none` under
`buffer` is an error the type could have prevented.

The resolution is to attach it to the two methods that actually read it:

```wit
attention:     func(rs: list<borrow<rs-working-set>>) -> result<_, error>;
fold:          func() -> result<_, error>;
buffer:        func(start-token: u32, geom: rs-geometry) -> result<_, error>;
fold-buffered: func(lens: list<u32>, geom: rs-geometry) -> result<_, error>;
```

No option, no dummy: unconditional where it is meaningful, absent where it is
not. The FOLDED slot is unaffected either way — it comes from the working-set
handle, never from a geometry record.

**Its fields now exist.** When D9 was written, RS had no descriptor-port
family: `rs_buffer_slot_ids` / `rs_buffer_slot_indptr` were derived host-side
by `RsStore`. The commit "feat(ptir): add the recurrent-state buffered-slot
descriptor port family" added `Port::{RsBufferPages, RsBufferIndptr,
RsBufferLen, RsWSlot, RsWOff}` at registry tags 10–14 (wire-additive), the
`PIE_DEVICE_PORT_RS_*` capability bits, the `FireGeometry` fields, and the CUDA
`descriptor_resolve` cases. `rs-geometry`'s five channels lower to exactly
those ports.

> **Lowered as of §10.1.** When this section was written, `rs-geometry` was
> RECORDED AND VALIDATED but not LOWERED: the `rs_translation` segment and its
> application in `batch_compose` did not exist, so a buffered fire still went
> through the host-derived `RsStore` path. Both now exist, the ports are
> claimed by the SDK and validated by the engine, and Metal mirrors the
> slices. Keeping the record's field set right BEFORE anything depended on it
> was the point — D8 chose records, and adding a record field later is
> breaking.

### 9.4 One host implementation, one SDK core

WIT duplication across the three interfaces is a statement about the GUEST
surface, not a mandate to fork the implementation. Both sides collapse it
immediately:

- **Host**: all three interfaces map `forward-pass` to the same Rust
  `ForwardPass`; the three `HostForwardPass` impls are thin delegations to
  `ProcessCtx::core_*`. The interface-selection check (`core_gate`) lands on
  the first state-binding call, because `constructor()` is infallible in WIT.
- **SDK**: three newtypes over one private `PassCore`, whose `wit` field is a
  three-way enum. The newtypes are what make a cross-kind helper a COMPILE
  error; everything below them exists once.

### 9.5 The hybrid `attention` call is deferred to first submit

`forward-hybrid.attention` binds KV and RS in ONE call (D7). The SDK still
exposes them as two independent, order-free calls (`attention` and
`recurrent`), recording both and issuing the single WIT call from
`flush_state_binding` at the top of `attach_program`. Guests never have to know
that the two halves travel together.

### 9.6 `option<kv-binding>` is accepted but not yet implemented

The hybrid `kv: none` arm — a `fold-buffered` fire running recurrent layers
only — is rejected by the host with an explicit message. `BoundForwardPass`
still requires a KV working set, and today a `fold-buffered` fire binds KV like
any other fire, so nothing regresses. The WIT admits `none` now so that wiring
it later is additive rather than breaking.

### 9.7 The split immediately caught two polymorphic inferlets

`beam-search` and `text-completion-bench` branched at RUNTIME on
`rs_state_size() > 0` to drive both a pure-attention model and a hybrid GDN
model from one body. With `ForwardPass` split three ways that polymorphism has
to move to compile time.

Both now branch at the top on `model.pass_kind()` over two monomorphisations.
The body is written once inside a `macro_rules!` and expanded twice, so the two
versions cannot drift. This is the cost the split was always going to impose,
and it is the correct place to pay it: the alternative — a shared trait over
the common subset — would reintroduce exactly the cross-kind helper the split
exists to forbid.

`mtp-native-verify` and `direct-channel-e2e` turned out NOT to be polymorphic
(they construct an `RsWorkingSet` unconditionally) and moved straight to the
hybrid interface.

### 9.8 Acceptance criterion met

`trackb-h2o`, `trackb-snapkv`, `tova-attention`, and `quest-attention` now
import `ptir::attention::prelude` and can no longer name a hybrid model's pass
type. There is deliberately no unprefixed `ptir::prelude`.

---

## 10. Remaining work

A, B, and C have landed. What is left, in the order it should land.

### 10.1 Lower `rs-geometry` (the other half of B0) — LANDED

The surface was final; the plumbing was not. The five RS ports
(`Port::{RsBufferPages,RsBufferIndptr,RsBufferLen,RsWSlot,RsWOff}`, tags 10-14)
existed and CUDA `descriptor_resolve.hpp` resolved them into
`FireGeometry::rs_buffer_*`, but **nothing anywhere read the result**. The
descriptor-resolved RS family was write-only. That is now closed.

**The translation is per-`(program, request)`, not per-program.** This is the
one place the RS path cannot mirror KV. A pass has ONE KV working set, so
`kv_translation_indptr` has `n_prog + 1` entries. A pass has one RS working set
per REQUEST, so `rs_translation_indptr` has `Σ_p R_p + 1`. That is also why the
translation is applied in `batch_compose::append_rs` rather than beside the KV
translation in `dispatch.cu` — the device-geometry loop there does not have
per-program request offsets in scope, and `append_rs` does.

**There is no masked-read escape for RS.** `translate_resolved_page_ids` maps an
out-of-range KV read page to 0 when a mask will discard it anyway. The RS twin,
`translate_resolved_rs_slot_ids`, refuses: out-of-range or unmapped is always an
error. A KV page the mask discards costs nothing to misread; a buffered
activation is folded into the state, where a wrong value is indistinguishable
from a right one and is never recoverable.

What landed, layer by layer:

- Engine: `RsStore::buffer_translation` (dense WS-relative buffer page →
  physical slot, `RS_TRANSLATION_UNMAPPED` for holes); `PreparedRs.translation`
  built at the *end* of `prepare_many_impl`, after `publish_batch`, so it
  reflects the pages this fire materialized or copied-on-write.
- Wire: composition at both the single-row and multi-row merge sites in
  `scheduler/wire.rs`, mirroring `rs_buffer_slot_indptr` exactly.
- ABI: `LaunchPlan` + `PieStepLaunchDesc` + `LaunchView` + `step_launch.hpp`,
  with CSR validation in `abi_validation.hpp`. **`PIE_DRIVER_ABI_VERSION`
  19 → 20**; `pie_driver_abi.h` regenerated.
- CUDA: `batch_compose.hpp` now prefers `geom.rs_buffer_slot_ids` for
  device-geometry programs and translates per request; `append_rs` returns
  `bool` so a bad translation fails the compose instead of silently producing
  a wrong slot.
- Engine `core_program`: the five RS ports are validated as **optional**
  bindings. A pass always binds `rs-geometry`, but a program only traces the
  buffer addressing when it actually addresses the buffer, so absent-but-
  attached is legal here and remains an error for the KV family.
- SDK: `bind_recurrent` claims the five ports — but only for an explicitly
  supplied geometry, never for the synthesized `rs_geometry_fold_all()`
  default, which would otherwise put five dead ports in every plain recurrent
  trace.
- Metal: mirrors the two new slices as pass-through.

**Open along the way:** whether `PortSource::Const` needs an RS analogue, and
whether the host-derived and channel-driven RS paths should be made mutually
exclusive per fire rather than merely unused-in-parallel.

### 10.2 E — Python / JavaScript SDKs

Larger than §7 implies. Both target an older `pie:core/inference` surface
(`imports/inference.py`, `pie-core-inference.d.ts`) rather than
`pie:inferlet/forward`, and neither is in the `scripts/sync-wit.sh` set, so
they did not break with B and will not track future WIT changes automatically.
Bringing them onto the current surface is a port, not a rename.

### 10.3 D — per-stage PTIR containers  *(separate PR)*

Unchanged from §7 / §9.1. D4(a) hooks-as-methods and D4(b) per-stage containers
must land together; (a) alone would force the host to reassemble and
canonicalize containers, which is the thing D4 exists to avoid.

### 10.4 Loose ends

- `model::pass_kind()` returns `recurrent` only when `kv_page_size() == 0`,
  which no registered model satisfies. That arm is unreachable and therefore
  untested, and `forward-recurrent.wit` has no in-tree consumer.
- `forward-recurrent.wit` documents `fold-buffered` as computing no logits, but
  `gdn-foldcommit` attaches an epilogue that reads `logits` and passes. One of
  the two is wrong; the doc is the more likely candidate.
- §8.2's Metal RS flag-bit collapse
  (`driver/metal/src/batch/batch_schedule.hpp:114` misreads `RS_FLAG_FOLD` as
  RESET) is pre-existing and still open.

---

## 11. The boundary model — ratified after Step B, supersedes §3.3/§3.4 fold modes

Step B shipped the fold modes as three sibling methods because that is the
shape the *implementation* had. Reviewing the RS cache's actual usage
afterwards showed the shape the *concept* has is different, and simpler. This
section records the corrected model and the surface it implies. It supersedes
the `fold` / `buffer` / `fold-buffered` triple in §3.3 and §3.4.

### 11.1 What the RS buffer is

A linear model's context is two adjacent spans, not one object:

```
[0 .......... F) [F .......... F+B)
 folded state     buffer
 compressed       uncompressed
 frozen           mutable
 O(1) memory      O(B) memory
 O(1) per token   O(B) replay per fire
```

The buffer holds each token's **pre-recurrence in-projection activations**
(`[mixed_qkv|a|b]`, per linear layer) — the recurrence's *inputs*, not its
state. So a buffered token still has an individual identity: it can be
addressed, replayed, discarded, reordered. Its storage layout is already
KV-shaped — page-major slabs, and `rs-buffer-page-size` is `kv-page-size` in v1.

Reading the recurrent state means: start from `folded`, scan the buffer.
`fold(n)` means: move `F` right by `n`.

**`fold` is semantically a no-op.** It changes memory and cost, never output.
It is a performance decision, legal exactly when the tokens it absorbs will
never be modified again — and *the guest is the only party that knows that*.
That single sentence is the whole linear-state programming model, and it is
why this is a guest-facing API rather than a runtime heuristic.

Corollaries:

- **Never folding must be valid and correct.** A linear model run purely from
  the buffer is O(n) memory and O(n) per fire — pathological, but it must
  produce identical output. Fold is opt-in.
- **Buffer ops are KV ops.** `discard` / `slice` / `copy-into` / `reorder` have
  the same meaning on the buffer that they have on a KV cache, including being
  the same kind of *approximation* (the buffer is a scan, so dropping an
  interior token yields "the sequence without it", exactly as KV eviction
  does). Only `fork` and `reset` are possible on the folded span.
- **Observation belongs at the buffer.** Once folded there is nothing
  per-token left to observe, so `on-recurrent` and any recurrent intrinsic tap
  the replay. §10's Tier 2 therefore depends on the read path below.
- **Fold is what enables sharing.** A content-addressed RS prefix requires the
  prefix to be exactly one compressed object. Fold is the *producer* of
  shareable state, not an obstacle to programmability.

### 11.2 The blocking defect: the buffer has no read path

The mental model above is **not** what is implemented. There are exactly two
buffer operations in the driver
(`driver/cuda/src/batch/forward.hpp:161-173`), and `rs_buffer_slab()` is
called from exactly two sites per model file:

- `rs_buffer_write` — scatter in-proj activations into slabs, `write_state`
  forced false (`qwen3_5_forward.cpp:510`).
- `rs_buffer_fold` — gather from slabs and fold `commit_len[r]` tokens into
  `recurrent_state[slot]`.

There is **no third site**. Nothing ever reads the buffer *as state*. Every
recurrence initializes from `recurrent_state[slot]`, i.e. from the folded
boundary. So a `buffer` fire actually means: advance transiently from the
folded state over this fire's own tokens, emit logits, **discard** the advanced
state, and stash the activations for one later fold.

That is a staging area for a deferred fold, not a token-granular store.

Consequences:

1. A single `buffer` fire looks correct because its tokens are inside its own
   span. **A second `buffer` fire on a non-empty buffer is wrong** — it
   restarts from the folded boundary and cannot see the first chunk.
   `gdn-foldcommit` buffers exactly one chunk at `start = 0`, so multi-chunk
   accumulation has never executed in-tree.
2. `buffer(start-token)` therefore does not merely carry a redundant argument
   (§9.2): it *invites* the broken case. Under §11.3 it disappears.

The storage layer is closer than this implies. The layout is already
token-granular; `RsStore::resolve_buffer(ws, start, len)` already resolves
arbitrary token ranges and has **no production caller** — only tests, which
already cover non-prefix ranges (`store/rs/tests.rs:333`); and "advance without
persisting" already exists as `write_state = false`. The missing read is a
recombination of parts that exist: gather from slabs, advance a *scratch*
state rather than the committed slot, continue into the fire's own tokens, do
not persist.

### 11.3 The surface: the fold modes disappear into the state binding

A fire running `[P, P+T)` sees one contiguous uncommitted span
`[F, P+T)` = `[buffer | this fire's tokens]`. All it declares is where `F`
ends up. The three Step B modes are three values of that one scalar:

| Step B | new boundary | `fold-len` |
|---|---|---|
| `fold` | `P+T` | everything |
| `buffer(start)` | `F` | 0 |
| `fold-buffered(n)` | `F+n` | n |

An open enumeration of *kinds* collapsed into a closed scalar over
*positions*. That is what makes the mode axis safe to fold into a record: a
variant has cases to add, a number does not. It is the same move as folding
the KV/RS geometry variants into parameters (D8).

Once the read path exists, buffer addressing is an input to the **recurrence**,
not to the fold decision — you cannot initialize the recurrence without knowing
which slabs to replay, and that is true of every fire, not of two of them. So
the geometry belongs on the state binding, exactly as `kv-geometry` does:

```wit
record rs-geometry {
    /// How far the folded boundary advances, per request. The twin of
    /// `kv-geometry.kv-len`: 0 = pure buffering, `buffer-len + fire tokens` =
    /// fold everything. Counts over [buffer | this fire's tokens].
    fold-len:        borrow<channel>,
    /// Buffer pages the replay reads.
    readable-buffer: page-span,
    /// Buffer pages an append may target.
    writable-buffer: page-span,
    buffer-len:      borrow<channel>,
    buffer-pages:    borrow<channel>,
    buffer-indptr:   borrow<channel>,
    w-slot:          borrow<channel>,
    w-off:           borrow<channel>,
}

resource forward-pass {
    /// State binding. The working set is the state's IDENTITY; the geometry is
    /// where it lives for this fire and where its boundary lands -- the same
    /// division as `attention(kv, kv-geometry)`.
    attention: func(
        rs: list<borrow<rs-working-set>>,
        geom: rs-geometry,
    ) -> result<_, error>;

    // No fold / buffer / fold-buffered. There is nothing left for them to say.
}
```

**This reverses §9.3, on a changed premise rather than a changed opinion.**
§9.3 moved the geometry off the binding because the buffer was exceptional —
touched by two calls out of many — so requiring it everywhere would have forced
a dummy. §11.1 says the buffer is half the state. Under that premise the
conclusion flips, and the plain-`fold` case stops being a dummy: an empty
buffer with `fold-len = T` is *degenerate but meaningful*, not meaningless. The
rule from §9.3 is unchanged — geometry goes where it is read — only the answer
to "where is it read" moved.

**`start-token` disappears.** New tokens always append at the buffer tail,
which the runtime already knows. Step B let the guest state a value the runtime
owns, which is why multi-chunk accumulation was expressible and wrong; under
the boundary model multi-chunk is just consecutive fires with no special case.

**The fast path is an SDK concern, not a WIT one.** A plain prefill/decode
still has to name eight fields whose values are constant and mostly empty.
Those lower through `PortSource::Const` into the cached container rather than
per-fire channel traffic, so the cost is at trace time; the ergonomics are
handled one layer up, where the SDK already defaults geometry (`fwd.attention(
&ws, .., .., &kv_len, ..)` — the `..` are defaulted page-spans). WIT stays
orthogonal and complete; the SDK supplies "fold everything, buffer empty" as
the default.

### 11.4 `fold-len` is geometry, and the price is a device-resident boundary

`fold-len` is structurally `kv-len`'s twin: per-request, per-fire,
device-resolvable, and already adjacent to the buffer CSR in the driver as
`rs_fold_lens_d`. The precedent is exact — `kv-len` is **already permitted to
be device-resident**: the decode-envelope path is selected by
`puts_channel(kv_len_channel)` (`pipeline/fire/geometry.rs:130`), i.e. by a
stage computing it with `ChanPut`, after which the host never learns the value
and `descriptor_resolve` consumes it on device.

That is the point of making it a channel. A speculative decode computes its
accepted count on device; with a host `u32` (Step B's `fold-buffered(lens)`)
the guest must round-trip it before issuing the commit fire — precisely the
round-trip channels exist to remove. As a channel, **verify and commit fuse
into one fire**:

```rust
fwd.epilogue(move || { n_acc_ch.put(&reduce_sum(cumprod(hit))); });
fwd.attention(&[&rs], &rs_geometry_with(&n_acc_ch, ..))?;
```

This is also the real fix for §11.6: today the correct shape (fold-commit) is
slower and more verbose than the incorrect one, so nobody writes it. Here the
fused form is both the fastest and the only correct one. The incentive aligns.

**The price.** `kv-len` is safe to be device-resident because it is
*descriptive* — the host reserved a page superset, and `kv-len` only says how
much to attend, so host ignorance costs nothing. `fold-len` is *imperative and
irreversible*, and today the host uses it for three things (`store/rs.rs`
module doc): `validate_fold` (granularity/capacity), publishing the boundary
advance in submission order (the basis of run-ahead safety), and retiring
fully covered head slabs.

Adopting the channel therefore commits to **the folded boundary being
device-resident state of the working set**, under the discipline the KV side
already follows: *the host must never need the value for correctness.*

- The host retains slabs conservatively and never retires on a device value.
  Reclamation stays with the guest's explicit `free-buffer` — which is already
  how it works (`gdn-foldcommit` calls it, and only the guest knows when).
- `validate_fold` demotes to a device-side clamp (`min(fold_len, buffer_len)`).
- The host tracks an upper bound on `F`; the exact position lives on device.
  Exactly the KV arrangement, where the host knows the page reservation and
  not `kv-len`.

**Incremental path.** Add the field now — D8 chose records and adding a record
field later is breaking, the same argument as §9.3 — but support only
trace-known constant channels at first. The host reads the constant at prepare
time and drives today's path unchanged; device-computed values open when the
store can carry the boundary. The WIT changes once and no guest is rewritten.

### 11.5 Ordering consequence for §10

§10.1 said the surface was final and only lowering remained. §11.3 changes the
surface, so §10.1's engine/SDK port wiring should land *against this shape*,
not Step B's — and it now wires ports for a record reached through `attention`,
not through two fold-mode methods. And the roadmap gains a tier ahead of observation:

**Tier 1.5 — the buffer read path.** Initialize the recurrence from
`folded state + replay(buffer)`. Without it the buffer is single-use staging,
"never fold" is not a correct steady state, and Tiers 2 and 3 have nothing to
attach to. It also resolves the §10.4 `fold-buffered` logits discrepancy: once
a fire can read the buffer, producing normal output without folding is the
ordinary case rather than a documented impossibility.

Tier 1.5 is larger than the read itself, because §11.4 moves boundary
ownership from host to device.

### 11.6 `mtp-native-verify` folds rejected tokens

Found while auditing RS usage. Of the RS consumers, only `gdn-foldcommit`
touches the buffer; `generate-gdn{,-n1,-frame}`, `beam-search` (which rolls
back with `fork` instead) and `mtp-native-verify` all use plain `fold`.

`mtp-native-verify` is a speculative-decoding inferlet that runs its `k+1`-row
verify fire under plain `fold` (`:209`) and only afterwards computes how many
rows were accepted (`:280`). The rejected tail is irreversibly absorbed before
the rejection is known. It passes today only because it has no `is-linear()`
guard and `rs-state-size` is 0 on attention models, making the binding a
no-op — but Step B moved it to `ptir::hybrid` because it constructs an
`RsWorkingSet` unconditionally.

The per-kind split does not catch this: it blocks *attention algorithms on
hybrid models*, not *fold misuse within a hybrid pass*. `fold` and `buffered`
are two methods on one type, so there is nothing for the compiler to see.
Fixing it is the acceptance test for Tier 1.5 — and today `gdn-foldcommit` is
the *only* coverage of the buffer path at all.

### 11.7 Implementation log — the boundary landed

`rs-geometry` now carries `fold-len` as its first field and rides on
`attention`; `fold` / `buffer` / `fold-buffered` are gone from both the
recurrent and hybrid interfaces, and `start-token` with them (new tokens append
at the buffer tail, which the runtime already tracks).

**`fold-len` is clamped to the tail.** Without a clamp, "fold everything" would
be `buffer-len + this fire's token count` — a value that changes per fire, so a
run-ahead decode loop could not state it once at bind time. Clamping makes
`u32::MAX` mean "through the tail", which is what the SDK's plain
`recurrent(&[..])` supplies.

**The host classifies, it does not compute.** `fire::rs_plan_for` reads the
constant at bind time and maps three positions onto the three `RsPlan` shapes
the driver already implements, so behaviour is byte-identical:

| `fold-len` | buffer | plan |
|---|---|---|
| `> 0` | empty | `Fold` |
| `0` | any | `Buffer { start_token = tail }` |
| `0 < n <= B` | non-empty | `FoldBuffered { n }` |

Everything else is a hard error, not a fallback: a boundary at or past the
buffer tail while the buffer is non-empty needs the read path (§11.2), a mixed
batch needs a per-row plan, and a device-computed `fold-len` needs the
device-resident boundary (§11.4). Approximating a fold in either direction is
unrecoverable — folding early destroys tokens the guest still wanted, folding
late silently drops them from the context — so refusing is the only honest
answer.

**Buffer occupancy is an upper bound.** `RsStore` tracks buffered *pages*; the
live token count rides on the `buffer-len` channel, which the host may not be
able to read. `rs_plan_for` therefore works with `pages * buffer-page-size`.
That is sufficient because the classification only asks whether the buffer is
empty and whether a boundary lands inside it — never for an exact length.

**SDK.** `recurrent(&[&rs])` (and `recurrent::attention`) keeps its signature
and synthesizes the fold-everything geometry, so the plain inferlets are
untouched; `recurrent_with(..)` / `attention_with(..)` state the boundary
explicitly. `gdn-foldcommit` is the only rewrite: `fold_len = 0` to speculate,
`fold_len = accepted` to commit.
