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

### 10.2 The buffer read path — LANDED

Every position `rs_plan_for` refuses needs the SAME missing primitive: a
recurrence that starts from `folded ⊕ replay(buffer)` rather than from
`folded`. There is no such read path — every recurrence initializes from
`recurrent_state[slot]`, and `driver/cuda/src/batch/forward.hpp` defines
exactly two buffer ops, `rs_buffer_write` (scatter in-proj activations) and
`rs_buffer_fold` (gather them back and fold).

**One of those positions was not refused — it was silently wrong.** `fold-len
= 0` onto a NON-EMPTY buffer classified as `Buffer` and ran: the new tokens'
recurrence started from the state at `F`, ignoring the `B` tokens already
buffered, and the fire emitted logits anyway. A wrong answer with no symptom.
It is now refused. (`start_token` was independently wrong for that case too:
it was `buffered[0]`, a whole-page UPPER BOUND that is exact only when the
buffer is empty. With the case gone, the planner emits `start_token = 0`
always; `RsStore` keeps the general offset because the store was never the
part that could not do this.)

The cost is honest: a speculative chunk must now fit in one fire. That is a
real limitation, but it is one the guest can see.

**What the remaining work actually is — much less than it looked.** The
chunked FLA kernel ALREADY takes `commit_len`, and already means by it "scan
`N` tokens, but persist the state after only the first `n`"
(`launch_chunk_gated_delta_prefill_batched_state_bf16`, threaded from the
commit-advance replay). The primitive for a boundary strictly inside a scan
therefore exists. What is missing is only the **token layout**: the linear
layers must process `B_r + T_r` rows per request, with rows `[0, B_r)` gathered
from the buffer slabs (no in-proj — the replay path already skips the GEMM
entirely, so the buffered prefix needs no hidden states) and rows
`[B_r, B_r + T_r)` produced by in-proj as usual. Then:

- `commit_len[r] = n_r` in the extended layout gives every fold position,
  including `n == B + T` (the fast path) and `B < n < B + T`.
- Outputs are the last `T_r` rows of `core_out`; the early
  `if (commit_len != nullptr) return;` at `qwen3_5_forward.cpp:1206` becomes a
  slice instead of a bail-out.
- `fold-len = 0` on a non-empty buffer needs no scratch state after all: give
  the row a copy-on-write folded slot and simply do not publish it. `RsStore`
  already allocates CoW slots for `write_state = true`; a scratch state is
  just an unpublished one, so the driver needs no notion of "scratch".

Concretely that is: extended `qo_indptr` for the linear layers only, an offset
on the in-proj GEMM output, conv-state handling over the extended layout,
workspace sized for `B + T`, and per-row `B` on the wire — duplicated across
`qwen3_5_forward.cpp` and `qwen3_5_moe_forward.cpp`.

**Why it was not landed at the time.** It is a numerics change to the FLA/conv
indexing with no way to validate it in this environment: there were no model
weights on the box, so nothing could run the kernel end to end. Landing several
hundred lines of unvalidatable indexing changes into the one path where a
wrong value is unrecoverable is worse than leaving the positions refused. The
analysis above was the deliverable; `gdn-foldcommit` extended to two chunks is
the test that must accompany the implementation.

#### 10.2.1 The read path, as built — LANDED

A GPU is available now, so it landed against real weights.
`cuda_gdn_foldcommit::two_chunks_need_the_buffer_read_path` flipped from
asserting the refusal to asserting `chained=ok`.

**The layout IS the buffer token space.** For request `r` with `B_r` buffered
and `T_r` new tokens, the linear layers run over `E_r = B_r + T_r` rows, and
extended row index equals buffer token index. That one identity makes the
gather, the scatter and the conv/FLA windows all fall out of the same
`qo_ext`, and it is why nothing needed a new kernel:

- `qo_ext[r+1] = qo_ext[r] + B_r + T_r`, built ONCE per fire (every linear
  layer runs the same rows) and uploaded into `Qwen3_5LinearAttnWorkspace::qo_ext`.
- Inside `linear_attn_layer_body` the parameters are renamed `N_new` /
  `qo_new_*` and `N` / `qo_indptr_*` are **rebound** to the extended space, so
  conv, prep and every FLA variant inherit it without a single edit. Only the
  three places that must stay in the fire's own token space name `N_new`: the
  in-projection source, `z`, and the post-recurrence epilogue.
- In-proj becomes one GEMM per request when a read is present, writing at
  `qo_ext[r] + B_r`; with no read it is the same single full-width GEMM as
  before, byte for byte.
- The gather mirrors the fold gather, but off the **read** CSR: reads cover
  the whole live buffer from page 0, writes only the span this fire appends.
  They are different intents on the same pages, which is why the wire carries
  both rather than widening one.
- The scatter had a latent assumption that the write span starts at buffer
  token 0. It does not any more: `page_span` starts at the page CONTAINING
  `B_r`, so the first listed page can begin before the appended span. It now
  clips per page and offsets into the slab.
- The epilogue slices rows `[qo_ext[r]+B_r, qo_ext[r+1])` back down to
  `[qo_new[r], ...)`. That shift overlaps itself, so it cannot be an in-place
  memcpy — it lands in `v_fp32`, which has the identical shape and is dead the
  moment the FLA returns. No allocation for a path most fires never take.
- `linear_decode` is forced off when a read is present: those kernels assume
  `N == R`, which an extended layout breaks.
- `B + T > max_tokens` is refused explicitly; every LA scratch buffer is sized
  to `max_tokens` and the replayed prefix consumes rows out of exactly those.

**What did NOT change.** A fold still carries no read (`FoldBuffered` gathers
from page 0 and its `start` is 0, so `buffer_read_lens` is 0 for it), and the
"a fold fire produces no usable logits" rule is untouched — see §10.8. An
all-zero read side is cleared on the wire, so the overwhelmingly common
empty-buffer fire is bit-identical to before.

**Metal refuses it.** Metal has no extended layout and would run the new
tokens from the folded state, silently ignoring the buffer — the exact failure
this design exists to prevent. `batch/compose.cpp` throws instead.

Still refused, and still needing work: a fold boundary strictly INSIDE the new
tokens (needs `commit_len` expressed over the extended layout), and a fire
whose rows disagree about where the boundary lands.

#### 10.2.2 Logical vs physical buffer tokens — the `buffer_head` — LANDED

Landing the read path exposed a deeper defect it had been standing on. A fold
absorbs `n` buffered tokens, but a buffer PAGE holds `buffer_page_tokens` of
them, and `bootstrap.rs` sets `fold_granularity: 1` while
`buffer_page_size = kv_page_size`. A fold therefore lands MID-PAGE as the
normal case, not an edge case. `advance_fold` released only the whole covered
pages and decremented the fill — but the survivors kept their in-page offsets
and nothing recorded that. Every buffer span was then computed as though
logical token 0 still lived at physical offset 0, so:

- the read gather REPLAYED tokens the fold had already absorbed;
- the write scatter OVERWROTE live buffered tokens;
- a second consecutive fold gathered from physical 0 — a bug that predates the
  read path entirely.

The fix names the thing that was missing. `RsEntry.buffer_head` is the physical
offset of LOGICAL buffer token 0, always `< buffer_page_tokens`; logical `k`
lives at physical `head + k`. `page_span` resolves it host-side so the CSRs
start at the page CONTAINING the span, and the head travels per row on the wire
(`rs_buffer_heads`, ABI 21 -> 22) because the driver's gathers and scatters
must convert too:

```
start      = head + <logical start>       // fold/read: +0; write: +B
page_first = (start / page) * page        // the CSR's first page
phys0      = max(page_tok0, start)
in_page    = phys0 - page_tok0            // offset INTO the slab
tok0       = phys0 - head                 // back to LOGICAL -> extended row
```

Two rebases keep the head from ratcheting. A fold that empties the buffer, and
a `free_buffer` that removes page 0 or empties the buffer, both reset it to 0:
there is no survivor to hold in place, and the next append should start at
physical 0. Without them `mtp-native-verify` — which frees all its slabs every
window — walks the head up until a fresh one-page window cannot hold its own
`k+1` tokens.

A fold's gather span is a READ, not an append, so `publish_prevalidated` no
longer adds it to `buffer_fill`; it had been re-adding what `advance_fold` had
just subtracted, visible whenever a fold took more than half the buffer.

Metal refuses a non-zero head for the same reason it refuses a non-zero read:
its loops are still logical, and a wrong recurrent state gets folded and cannot
be recovered.

### 10.2.3 One fire can append AND fold — LANDED

Until now a fire either wrote its tokens to the buffer or folded them, never
both. `rs_plan_for` refused `fold-len == b + t` whenever `b != 0`, telling the
guest to "fold and buffer in separate fires". That refusal cost a whole extra
forward pass for the commonest speculative shape there is: buffer a draft,
then accept it and keep going.

It turns out to need no kernel work at all. The extended layout `[b | t]` IS
the row's buffer token space, so when the fold takes the WHOLE extended row
the boundary is the LAST extended token — exactly where the FLA's ordinary
end-of-sequence state writeback already lands. The change is one bit of
routing: `write_state` was being forced false for every buffered pass (a
buffered pass must not disturb the folded state); it now stays true when the
pass also folds.

`RsPlan::Buffer` therefore carries `fold_tokens` per row, and the planner's
classification runs in this order — the order matters:

```
n == 0            -> Buffer   pure append; extended layout when b > 0
n == b + t, b==0  -> Fold     plain in-forward advance, no buffer at all
n == b + t, b!=0  -> Buffer   write-and-fold
n <= b            -> Commit   FoldBuffered; the rows ARE the replay
otherwise         -> REFUSED  boundary strictly inside the new tokens
```

`n == 0` has to be tested first: with `b > 0` it otherwise falls into `n <= b`
and calls `validate_fold(0)`, which rejects a zero-token fold.

The store needed the same unbundling. `RsPreparedWrite.buffer_span` now
carries an explicit `RsBufferIntent`: a `Write` span bumps `buffer_fill`, a
`Replay` span does not. The old heuristic — "a span on a folding row must be a
replay" — is exactly what a write-and-fold breaks. `publish_prevalidated` also
had to be reordered to write, then count, then `advance_fold`, since the fold's
arithmetic is over the POST-write buffer.

On the wire, `RS_FLAG_FOLD` can no longer discriminate: a pure commit and a
write-and-fold both set it but take opposite driver paths (gather-from-slabs
vs. extended layout). A new orthogonal bit `PIE_RS_FLAG_BUFFER_WRITE` marks
"this row's buffer span is a write", and `plan_rs_execution` gates its fold
detection on it. ABI 22 -> 23.

#### Why the interior boundary was refused — LANDED in §10.2.8

`b < n < b + t` — the boundary landing strictly INSIDE the fire's own new
tokens. This was the last refused RS position; it is now implemented, and the
sketch below is what got built. Kept here because the reasoning for rejecting
`commit_len` is the reason the shape looks the way it does.

The obvious implementation is `commit_len`, which the chunked FLA already
takes. It does not work: `causal_conv1d.cu` and `gated_delta_net.cu` both do
`if (c < Nr) Nr = c;`. `commit_len` TRUNCATES the sequence. That is right for
the commit-advance replay, which is state-only and discards its outputs, but a
fire folding at an interior boundary still owes logits for every one of its
tokens — and the tokens past the boundary would get none. Measured directly:
the fold landed on the right token, and the last token's logits were wrong.

The shape that would work is two FLA calls over a 2R-segment layout tiling the
extended array exactly — `[qo_ext[r], qo_ext[r]+n_r)` and
`[qo_ext[r]+n_r, qo_ext[r+1])` — the first with `slot_ids = [slot_r, -1, ...]`
and `write_state=true`, the second with `[-1, slot_r, ...]` and
`write_state=false`, chained on the same stream through the state the first
just wrote. Both kernels early-return on a negative slot and leave `out`
untouched, so each call fills its own half. It costs a second launch per layer
and has to be replicated across every FLA variant.

#### The read path had never actually run

Landing this exposed that §10.2 and §10.2.2 were dead code on the GPU. Two
independent gaps swallowed the read side between the wire and the kernels:

- `run_forward_dispatch` copied the buffer WRITE CSR into `ForwardInputs` but
  not `rs_buffer_read_*` or `rs_buffer_heads`, so `has_buffer_read` was always
  false and the extended layout was never built.
- `batch_compose` never carried the read side either. Every fire in this engine
  is descriptor-composed, and composition REORDERS requests (wire programs
  first, device-geometry ones after), so the read rows have to be permuted with
  everything else. They are host-derived and translation-free — a replay span
  is a property of the working set's occupancy, which a channel-resolved
  `rs-geometry` cannot name — but they still have to travel through `append_rs`.

The existing read-path tests passed because their fold shapes all routed to
`FoldBuffered`, which gathers through the WRITE CSR. Only a fire that writes
and reads in the same pass distinguishes them, which is what
`a_fire_can_append_to_a_buffer_and_fold_through_it` now does: it appends two
tokens onto a two-token buffer and folds all four, then compares against the
same four tokens folded two fires at a time — continuing BOTH arms one token
further, so agreement pins the folded state and not merely the logits along
the way.

### 10.2.4 RS reachability audit — LANDED

The read-path find in §10.2.3 was the second time an RS array had been built
correctly, validated correctly, and then silently dropped inside the driver
(§10.1 was the first: the descriptor-resolved RS family was write-only). Two
instances is a pattern, so the whole RS surface was walked end to end —
host build → wire → ABI → validation → composition → frame → the
`ForwardDispatchInputs`/`ForwardInputs` copy → both model forwards → use site.

Confirmed reachable, both dense and MoE: `rs_slot_ids`, `rs_slot_flags`,
`rs_fold_lens` (host and device), `rs_buffer_slot_ids`/`_indptr`,
`rs_buffer_read_*`, `rs_buffer_heads`, `rs_translation`/`_indptr` (consumed at
composition, by design), and the resolved `FireGeometry::rs_buffer_slot_ids`/
`_indptr`.

Two more gaps found and closed.

**The read side never reached a TP follower.** Only rank 0 sees the launch
descriptor; a follower recovers every per-request array by reading it back off
the device, so an array that is never STAGED to device never reaches it.
`rs_buffer_read_*` and `rs_buffer_heads` were host-only spans and had no
`PersistentInputs` home, no `TpBuf` id, and no broadcast. A TP follower would
therefore skip the replay of the buffered prefix and fold a DIFFERENT recurrent
state than rank 0 — and the per-layer all-reduce mixes the two without
complaint, so the symptom is degraded output, not a crash. They now stage,
broadcast, and read back exactly like `rs_buffer_slot_ids`, and the follower
throws if the reconstructed read CSR disagrees with its id count.

This box has one GPU, so the TP path is reasoned and pattern-matched rather
than executed. The consistency check is there because of that.

**`rs_w_slot` / `rs_w_off` were resolved and ignored.** `descriptor_resolve`
fills them from the channels and nothing read them. The driver's scatter walks
a row's listed slabs page-major and contiguously — it has no per-token
indirection — so the only descriptor it can honour is the one that says exactly
that, which is what every guest supplies today (`rs_span()` in
`mtp-native-verify` is literally `t / page`, `t % page`). Rather than keep
resolving a port and discarding it, `append_rs` now CHECKS the descriptor
against that layout and refuses anything else with an explicit "not
implemented". A silently ignored port and an honestly refused one are very
different promises to a guest.

**Still dropped, deliberately: `FireGeometry::rs_buffer_lens`.** The
per-request live buffered token count resolved from a channel is not read; the
host-derived value wins. That is exactly the inversion §11.4 (`t15`) exists to
fix — when the folded boundary becomes device-resident the DEVICE value has to
be authoritative — so it is left as the hook rather than half-wired now.

`ForwardInputs::rs_slot_flags_h` is also set by every caller and read by no
model. It is harmless (the flags are consumed in `frame.cpp` to derive
`is_fresh`), but it is dead storage of exactly the shape that hid the two real
bugs, and should go.

### 10.2.5 Mixed-position batches — LANDED

One fire may now land its rows' folded boundaries in DIFFERENT places: request
A appends to its buffer while request B folds outright. This is the last of the
"a fire folds uniformly today" refusals and the shape real serving wants — one
request committing while another speculates.

**Why it is nearly free.** A buffered append and a plain in-forward fold are
the SAME dispatch. Both run this fire's own tokens through the whole stack over
the extended `[buffered | new]` layout, from the same initial state, producing
the same outputs. They disagree about exactly one thing: whether the recurrence
PERSISTS at the end. So the fix is not a second pass — it is to demote
`write_state` from a per-pass boolean to a per-row mask.

`const std::uint8_t* write_state_mask` now rides alongside `bool write_state`
in every GDN and conv kernel; the gate is
`write_state && (mask == nullptr || mask[r] != 0)`. A null mask means the pass
is uniform, so every pre-existing call site is byte-identical to before, and
`write_state = false` (frozen verify) still vetoes at the pass level. There are
only six real gate sites across the two kernel files; the rest is signatures.

The mask is derived IN THE MODEL, from arrays it already has:

```
persists(r) = rs_fold_lens[r] != 0            // this row folds
           || buffer CSR span of r is empty   // this row owns no buffer
```

and uploaded into the linear-attention workspace next to `qo_ext`. Deriving it
rather than wiring a new per-request array is deliberate: `ForwardInputs`,
`run_forward_dispatch`'s copy block and the TP follower readback have now
swallowed a per-request array three times (§10.2.3, §10.2.4). An array that is
never plumbed cannot be dropped.

**A row that folds in-forward inside a buffered pass** carries
`start_tokens = row_tokens = fold_tokens = 0` and `RsPlan::Buffer::in_forward`.
It cannot carry a fold LENGTH: `validate_fold` measures against buffered
capacity, which such a row does not have. The driver recognises it by its empty
buffer CSR span — which also makes the buffered-write slab check skip it, and
makes the scatter loop skip it for free (the loop is bounded by the CSR).

**Three refusals were relaxed and one added.** `rs_plan_for` now classifies
each row independently and only requires uniformity for a pure COMMIT;
`plan_rs_execution` reads the write off the CSR span rather than the flag (
`PIE_RS_FLAG_BUFFER_WRITE` marks write-AND-fold, so a pure append carries no
flags at all) and refuses only a replay row sharing with a computing one; and
`LaunchGrouping` no longer sends every RS-buffer fire out alone —
`touches_rs_buffer` became `rs_batch_kind`, with `Solo` reserved for the pure
commit and a new rule that an RS-bound fire cannot share a wave with one that
binds none (the RS arrays are one-per-request; a partial batch does not
resolve). Metal, which has only the pass-level flag, now refuses a mixed pass
loudly instead of folding every row or none.

**A pure commit still cannot mix**, and this is not a limitation waiting to be
lifted: its rows are not computed at all. The linear layers gather activations
out of the slabs and return before the output projection. There is no per-row
switch that lets a computing row ride along.

**The acceptance test is one fire with two rows**, not two co-batched fires:
the scheduler admits at most one fire per instance per wave, so a single
inferlet cannot produce a mixed batch by submitting twice. `Duo` binds two RS
working sets and two disjoint page ranges of one KV working set, and
`gdn-foldcommit mixed` runs three identical duos — one buffering both rows, one
folding both rows, one mixed — comparing the mixed one against each uniform
reference row-wise. Two things make it non-vacuous:

- The observable is the PEAK LOGIT, not the greedy token. This model has strong
  attractors; the first attempt compared argmax and both references returned
  the same token from genuinely different states. The vacuity guard, which
  demands the two references disagree, is what caught it.
- Verified by negative control: forcing the mask pointer to null makes row 0
  report the FOLDING reference's value (16.50 instead of 14.19), so the test
  observes the mask and not merely that the fire ran.

### 10.2.6 `rs-geometry` rebalanced — the guest states positions, the runtime derives addresses — LANDED

`rs-geometry` had eight fields. Exactly ONE of them, `fold-len`, was a guest
decision. The other seven were buffer ADDRESSING, and the runtime computed
every one of them itself, from a store it is authoritative for, while the guest
was busy computing the same values and shipping them across the wire. Four
sites, all verified by reading before changing anything:

- `fire.rs:451` classifies rows from `store.buffer_tokens()`. The guest's
  `buffer-len` never reached the classifier.
- `rs.rs:411` builds the wire write CSR from `prepared.buffer_targets()` — the
  store's own allocation — not from the guest's `buffer-pages` / `indptr`.
- `rs.rs:432` rebuilds the read span from `store.buffer_head` plus the plan's
  start, under a comment that says outright it is doing so "rather than the
  mapping the guest saw".
- `batch_compose.hpp:731` does not USE `rs_w_slot` / `rs_w_off`. It checks that
  they describe the one contiguous page-major layout the recurrence scatters
  into, and REFUSES the fire if they do not.

So the guest's page arithmetic was a proof obligation with exactly one
satisfying assignment: get it right and nothing happens, get it wrong and the
fire is refused. `readable-buffer` and `writable-buffer` did not even reach
that bar — they were parsed into `RsGeometryBinding` and never read at all.

```wit
record rs-geometry {
    fold-len: borrow<channel>,   // the one decision: where the boundary lands
    buffer:   page-span,         // capacity grant, not an address
}
```

**Allocation stays on the API, addressing does not.** The distinction is
whether a wrong answer should FAIL or be FOUND. A fire that needs a buffer page
it was not granted must fail loudly, so the grant is guest-stated. Where within
that grant a token lands is not a decision at all — new tokens append at the
tail, and the runtime is the only party that knows where the tail is. Calling
this a rebalance rather than a simplification is the point: nothing moved to a
heuristic, because there was never a choice on that side to make.

**This is the same collapse §10.2 already performed on the other half.**
`fold` / `buffer` / `fold-buffered` were three values of one scalar and became
`fold-len`, a number over positions. The addressing half simply never got the
same treatment.

**It is a PREREQUISITE for §11.4 (`t15`), not a detour.** The guest derived
`w-slot` / `w-off` from a host-side `start`. t15's whole thesis is that the
host keeps only an UPPER BOUND on the folded boundary, which makes `start`
unknowable at trace time — so under the old record t15 was implementable only
if the guest stated a position it could not know. The record had to shrink
before the boundary could move.

**Cost.** Not one wire field changed: the runtime already derived what the
guest was sending, so the driver and the ABI are untouched. Registry tags 10-14
are RESERVED rather than reclaimed — renumbering would silently change the
meaning of already-compiled containers, a far worse trade than five unused
names. `RsBufferLen` is the exception and its DIRECTION inverts: the live
buffered token count is exactly what t15 makes device-resident, so it returns
not as something the guest states but as something the device writes and the
host reads as a bound. `FireGeometry::rs_buffer_lens` is already staged for it.

**What became inexpressible, honestly.** Per-token buffered-activation
targeting (already refused by the driver as "not implemented"), non-tail
placement (already closed by §11 as "expressible and wrong"), and a guest
under-reporting its own buffer occupancy (never consulted). The one real loss
is narrowing a replay to less than the row's occupancy — a partial or selective
fold. That is a meaningful algorithm, but it was never reachable through these
fields either, and when it arrives it should arrive as a SEMANTIC field beside
`fold-len` saying which tokens the fold absorbs, not as a page span that makes
the guest reconstruct the runtime's physical layout again.

**The guest code is the evidence.** Deleting the fields made `rs_page` — the
buffer page size — fall out of both test inferlets everywhere EXCEPT the
`alloc_buffer` / reserve calls. The only thing an author still needs the
physical page geometry for is the capacity grant, which is precisely the one
decision the record kept.

### 10.3 `mtp-native-verify` rewritten as fold-commit — LANDED

The Tier-1.5 acceptance test was broken in two independent ways, both of which
the boundary model makes obvious.

**It folded before it knew what was accepted.** The verify fire used the
default fold-everything geometry, so all `k+1` window tokens went irreversibly
into the recurrent state — and `n_acc` was computed from *that same fire's*
logits, afterwards. A rejected tail was already absorbed by the time it was
known to be rejected. This is precisely the situation the buffer exists to
prevent, in the one inferlet meant to demonstrate it. It now runs the two-fire
shape: verify with `fold_len = 0`, then commit with `fold_len = clen` once the
accepted length is known, then `free_buffer` for the tail.

**Its request layout could not exist on a linear model.** `bind_verify_rows`
built `k+1` REQUEST ROWS — a staircase where row `i` had `kv_len = seq_len+i`.
For attention that is equivalent to one causal row. For a linear model it is
not expressible at all: state is per-REQUEST, so `k+1` rows demand `k+1`
working sets holding divergent copies of one sequence's state, and
`validate_count` rejects binding one. It is now a single row of `k+1` causal
tokens, which is both equivalent for KV and the only shape with a single state
to advance.

**Two smaller bugs found while rewriting**, worth recording because both are
silent:

- The commit fire must re-supply the WINDOW's first `clen` tokens, not the
  `clen` tokens the verify predicted. The latter is the window shifted by one
  (each entry is what the target predicted *after* the corresponding window
  token), and the commit fire rewrites KV over the same span — so feeding it
  would have corrupted exactly the prefix being committed. The recurrent side
  never reads these tokens at all; they matter only to KV.
- The buffer must be re-reserved per window. Each window ends by freeing its
  slabs, and appending onto a buffer still holding the previous window's tail
  is now refused (§10.2) — correctly, since the recurrence would ignore it.

The test still cannot be RUN here (`bin/pie/tests/cuda_mtp_native_verify.rs` is
`--ignored` and needs Qwen3.5-0.8B weights), so this is a correctness rewrite
validated by construction and by matching `gdn-foldcommit`, which drives the
same two-fire shape.

### 10.2.7 The fold boundary can live on the device — LANDED

The last host round-trip in a speculative linear-model decode was not a
transfer of tokens. It was ONE NUMBER: the accepted count. The verify fire
computes it on device; the commit fire consumes it as `fold-len`. Between them
the host had to await, read the count back, and only then trace the commit —
serializing two fires that could have been enqueued together.

`fold-len` was already a channel, so the guest could always *hand over* a
device-computed value. What made it unusable is that the HOST plans the fire,
and the host's plan depends on where the boundary lands.

**The host keeps a bound, the driver keeps the value.** A row's fold length is
constrained to `[1, b]`, where `b` is its live buffer occupancy — a quantity
the host knows exactly at submission. So the host plans against `b` and the
wire slot carries `b`; a new per-row flag `PIE_RS_FLAG_FOLD_LEN_DEVICE` (ABI
23 -> 24) says "this is an UPPER BOUND, substitute the resolved value and clamp
it". The driver does exactly that, in `batch_compose::append_rs`, BEFORE
`plan_rs_execution` — so nothing downstream ever sees the placeholder and
`fold_qo_indptr` is shaped from the real length.

Carrying the bound rather than a sentinel is what keeps the change small. It
preserves the existing ABI invariant `folds == (fold_len != 0)`, it gives the
clamp something to clamp against, and a resolved `0` is refused outright — a
speculative commit folds at least the bonus token it is guaranteed to accept.

**Every value the device can name is the same dispatch.** The path initially
required the fire to carry no new tokens. It does not have to: because the
driver clamps to `b`, every reachable length lies inside the buffer, and every
such fold is the same `FoldBuffered` replay over slabs. The fire's own tokens
are as irrelevant here as they are to any commit. The price is that the
INTERIOR boundary (`b < n < b+t`) is structurally unreachable under a device
length — which costs nothing, since it is refused for host-known lengths too
(§10.2.3).

**The host must then stop believing it knows `F`.** This is the whole
correctness content. After a device fold the host cannot advance the boundary:
assuming MORE drops pages that are still live; assuming LESS replays tokens
already absorbed, which is a double fold and unrecoverable. So `advance_fold`
is SUPPRESSED, every page is retained, `buffer_fill` degrades to an upper
bound, and `RsStore::buffer_tokens` REFUSES
(`BufferOccupancyIndeterminate { bound }`) rather than returning a number that
might be wrong. `buffer_tokens_bound` and `buffer_tokens_exact` expose the
weaker fact for callers that can use it.

Indeterminacy is recoverable, not terminal: the guest's own `free_buffer`
empties the buffer and restores exactness. That is already the shape a
speculative loop has — buffer a window, commit its accepted prefix, free the
window, start the next. `mtp-native-verify` needs no restructuring to fit it.
A fork inherits the indeterminacy, since it shares the buffer.

**Why this needed the §10.2.6 rebalance first.** Under the old `rs-geometry`
the guest computed `w-slot`/`w-off` from a host-side `start`. If `F` is only
bounded, `start` is unknowable at trace time and that arithmetic becomes
unwritable. The rebalance deleted it, so there is nothing left for the guest to
have to lie about.

**Two things the driver had to learn.**

- **Wire programs can have ports.** `resolve_descriptors` only resolved ports
  for programs classified DEVICE-GEOMETRY; a commit fire is an ordinary
  host-geometry wire program with exactly one device-resolved number, so its
  port was never read and composition saw `has_rs_fold_len = 0`. There was
  already a precedent for the narrow case — an otherwise host-geometry
  attention pass gets its MASK resolved and nothing else — so `rs_fold_len`
  takes the same shape: `resolve_rs_fold_len` reads that one port and
  `is_device_geometry` stays `0`. Promoting the whole program would have been a
  far larger claim than the truth.
- **Port arrays are indexed by TAG.** `descriptor_resolve` sized its port
  arrays `[15]`. Since the rebalance reserved tags 10-14, "highest tag" is no
  longer "number of ports", and tag 15 wrote out of bounds. Now sized from
  `kPortRsFoldLen + 1` with an explicit guard.

**And one thing the GUEST had to learn: construction order.** A channel a later
pass consumes as a descriptor must be claimed by that pass BEFORE the producing
pass is submitted. Otherwise the producer infers it as a terminal host-read
output (`HostRole::Reader`) while the consumer declares `HostRole::None`, and
the registry rejects the conflicting re-bind. This is the SDK's existing F8
eager-claim rule; the device handoff is simply the first thing that depends on
it, so the test inferlet grew `Arm::build_fire`, which constructs without
submitting.

Metal refuses the flag. It fails in the quietest way of all there: with no
descriptor resolution, Metal would read a perfectly well-formed number — the
bound — and fold the entire buffer instead of the accepted prefix.

**Verified on GPU**: `cuda_gdn_foldcommit` 5/5, including the new
`the_fold_length_can_live_on_device`, whose device arm and constant-control arm
agree one token PAST the commit (so agreement pins the folded STATE, not the
logits along the way) while a third arm folding a DIFFERENT count genuinely
diverges — `a_next=271 b_next=271 c_next=2`. Also `mtp_logits_value_verify`,
`cuda_plain_gen`, engine lib 371 pass (2 known timing failures).

**A tolerance bug surfaced and was fixed on the way.** The mixed-position test
(§10.2.5) compared peak logits with an ABSOLUTE tolerance of `1e-2`. Its
reference arms are solo fires while the mixed arm is a two-row batch, so they
reduce in different orders — and at a peak logit of ~14 the bf16 ULP is
`0.0625`, six times the tolerance. The test was asserting a state divergence on
noise finer than the format can represent. The tolerance is now relative, and
the "the references must genuinely disagree" guard is measured on the same
scale so it stays meaningful.

**Still open**: the fully fused single-fire verify+commit of §11.4 needs the
interior fold boundary as well. Device-boundary alone delivers the smaller and
more valuable half — two fires, zero host round-trips between them.

#### 10.2.7a Wiring it into `mtp-native-verify`

§10.2.7 landed the capability against a purpose-built test. Nothing REAL used
it, so the speculative decoder that motivated the whole thing still round-
tripped its accepted count through the host. It no longer does.

`verify_window` became a non-submitting `build_verify`, whose epilogue also
publishes `n_acc + 1` into a `fold-len` channel; a new `build_commit` consumes
that channel as a descriptor. The two fires are now enqueued back to back with
nothing awaited between them. `commit_window`, which could not be traced until
`clen` was known, is deleted.

**The commit re-runs the FULL window, not the accepted prefix.** The prefix
length is not host-known, so it cannot appear in the KV geometry. That is free:
the verify fire already wrote KV for exactly that span, with exactly those
tokens, at exactly those positions, so re-writing it is bit-identical, and the
rejected tail is overwritten by the next window regardless. The recurrent side
ignores the tokens entirely and replays the buffered activations, folding only
as far as the device says. This is what makes a device-resident commit
possible at all.

**A compose-time instruction leaked into the kernels.**
`PIE_RS_FLAG_FOLD_LEN_DEVICE` is an instruction to `append_rs`: *substitute the
resolved port here*. Once substituted, the row is an ordinary buffered fold and
must be indistinguishable from one. But the flag byte was copied verbatim into
`out.rs_slot_flags` and travelled on into the frame and every kernel that reads
them. It is now cleared immediately after the insert. The general rule: a flag
that tells a compose pass what to DO is not a property of the composed row, and
must not survive the pass that consumes it.

#### 10.2.7b The equivalence test that cannot exist

The obvious test — decode the same prompt with a host fold length and a device
fold length and require identical output — is **not sound on this model**, and
believing it cost most of a session.

Three IDENTICAL host-mode launches, in ONE engine boot, against ONE model load,
produced three different token streams: `committed=23 steps=12`,
`committed=25 steps=13`, `committed=26 steps=9`. The cause is not the fold
path. The decode is greedy but the model sits in an attractor (its own drafts
come back as repeated `13477` / `3841`), so the top two logits are frequently
within a bf16 ULP; ordinary reduction-order variation flips an argmax; and
because the drafts FEED BACK into the next window, one flip forks the whole
trajectory. Every cross-run comparison in that regime is reading noise — it
will "confirm" a fix and then "regress" with no code change, which is exactly
what happened.

This generalises §10.2.5's lesson. There the fix was a relative tolerance on
peak logits. Here even that is unavailable, because the quantity under test is
a token stream, and a token stream downstream of a near-tie argmax has no
tolerance.

So the assertion is a **within-run invariant** instead. Each window the verify
epilogue publishes its committed length TWICE: once into the `fold-len`
descriptor the commit fire consumes, and once into a plain host-readable echo
channel. (Two channels, not one: a channel claimed as a descriptor declares
`HostRole::None` and so cannot also be a terminal host-read output — the F8
rule again.) The loop then checks, at every fold, that the number the driver
folded is the number the host itself derived from the sentinel tail, and fails
the inferlet on the first disagreement. The harness additionally requires that
every window was checked (`fold_len_checked == steps`) and that at least one
window folded more than the lone bonus token (`fold_len_nontrivial > 0`) —
without that second guard a driver that ignored the resolved value and folded a
constant `1` would pass, since most windows accept nothing.

The cross-mode EQUALITY claim keeps its home in `gdn-foldcommit::
the_fold_length_can_live_on_device`, which is a single fire over a fixed input
and is genuinely deterministic. That is the
right place for it: equivalence belongs at the primitive, not at the end of a
chaotic feedback loop.

#### 10.2.8 The interior fold boundary — LANDED

The last refused RS position, and the blocker for a single-fire speculative
verify+commit: a fold whose boundary lands strictly inside the tokens the fire
is itself computing.

**The driver shape is the 2R-segment sketch above, built with no kernel edits
at all.** Row `r` becomes two segments of one `qo_split` array — `[qo[r],
qo[r]+n_r)` and `[qo[r]+n_r, qo[r+1])` — dispatched twice on the same stream:
the HEAD with the tail slots negated and `write_state=true`, the TAIL with the
head slots negated and `write_state=false`. Every GDN and conv kernel already
early-returns on `slot < 0`, and the conv prefill already sources left context
out of the slot for `src_t < 0`, so the tail's convolution history is exactly
the trailing K-window the head persisted. The cut is invisible to the
arithmetic. A row with NO interior boundary sets `n_r = T_r`, leaving its tail
empty — so the 2R layout subsumes the 1R one and a mixed fire needs no special
case. The whole conv dispatch and the whole FLA cascade were wrapped in
lambdas whose parameters shadow the outer names, so neither body was edited.

No ABI change was needed. `fold_tokens[r]` already travelled as
`rs_fold_lens[r]` in extended (buffer-token) space; the driver had only ever
tested it for `!= 0`.

**The real defect was on the host, and it was older than this work.** The first
GPU run gave `outputs=ok states=BAD` — the tail's logits were right, so the
split itself was right, but the state the fire left behind was wrong. A
post-recurrence state probe (`PIE_RS_SPLIT_TRACE`) settled it in one run: the
folded state at the boundary matched every reference arm to 0.005%, and the
divergence was entirely in the NEXT fire, which drained the buffer.

`fire::rs` built the wire arrays AFTER `publish_batch`, and `publish_batch`
advanced the fold boundary. But every wire array describes the buffer as *this
fire's own rows are laid out* — extended row `j` is physical `buffer_head + j`,
the write CSR lists the pages that span covers, the read CSR starts at the
head. That is the PRE-fold frame. So a fire whose rows straddle the boundary
was handed a head, and a page list, from the far side of it: the interior fire
scattered its four tokens at physical 2..5 instead of 0..3, and the drain read
physical 0..1, which had never been written.

It survived this long because until now a fold either moved nothing (`n == 0`,
head unchanged) or emptied the buffer (`n == b + t`, which rebases the head to
0 anyway). Those are the only two cases where pre- and post-fold agree. The
fix is `RsStore::commit_folds`: `publish_batch` returns the fold advances as a
`#[must_use]` value and the caller applies them once the wire is built.

**It was corrupting the already-landed paths too.** With the head fixed, the
§10.2.7 device-fold test's negative control stopped differing — because that
control had been witnessing this same bug. A pure commit's gather had been
starting at the POST-fold head, so arms folding `n` tokens were folding the
`n` tokens starting at offset `n`. Its control now rests on a window of
DISTINCT tokens (it had been four copies of one token, where folding one
versus two is nearly unobservable) and on peak logits rather than the greedy
argmax — the same coarseness lesson as §10.2.5 and §10.2.7b, for the third
time.

**The test has no negative control, deliberately.** Folding IS a no-op: every
legal boundary must converge, so an arm that disagreed would be evidence
against the model rather than for the test. Four arms therefore land the same
context four ways — an interior fold at 2, an interior fold at 1, a reference
that folds before it buffers, and one that reaches the interior fire's exact
store state (`fill=4, head=2`) through an ordinary commit — and all four must
agree on both the tail's outputs and the state left behind. What gives that
teeth is that it caught the defect above: before the fix, `a_next=17.2500`
against `b_next=16.8750`, and the trusted-path arm was wrong too.

Metal refuses an interior boundary; it issues one call per fire and would fold
the whole row, leaving the host believing tokens are buffered that the device
has already absorbed.

### 10.4 E — Python / JavaScript SDKs

Larger than §7 implies, and larger than "a port" implies either. Three
compounding facts, in increasing order of severity:

1. **The interface they target no longer exists.** Not "is older" — `pie:core/
   inference` is gone from the tree; there is no `inference.wit` anywhere, and
   `world.wit` imports `forward` / `forward-recurrent` / `forward-hybrid`
   instead. `sdk/python/.../forward.py:22` still does
   `from wit_world.imports import inference as _inf`, and the JS side still
   ships `bindings/interfaces/pie-core-inference.d.ts`.

2. **The programming model changed, not just the names.** The current surface
   is PTIR: channels, traced stage closures, epilogues, per-architecture state
   bindings. The Rust SDK needs ~1950 lines of `ptir.rs` to express it. There
   is no mapping from `_inf.forward(...)` onto that — the Python and JS layers
   (`forward.py` 413 ln, `context.py` 441 ln, `generation.py` 590 ln, plus the
   generated bindings) have to be rewritten against a different shape, and the
   epilogue eDSL has no Python/JS equivalent at all today.

3. **Nothing would have told us.** Neither SDK is in `scripts/sync-wit.sh`, so
   they do not track WIT drift; and neither appears in any CI workflow — the
   only references are `release-pypi.yml` and `release-npm.yml`, which are
   manually triggered publishes. So these packages can be RELEASED in their
   current state, against an interface that does not exist. That is the part
   worth fixing first, and it is independent of the port: either gate the
   release workflows on a build, or mark the packages unsupported until the
   port lands.

Sequencing: (3) is a small, standalone change and should not wait for (1)/(2),
so it is DONE. `scripts/check-sdk-interfaces.sh` resolves the interface names
each SDK references against `interface/inferlet/`, and both release workflows
run it for their SDK package. It needs no componentize-py or jco — just the
names, which is what actually drifted. It currently fails, correctly:

- Python references `adapter`, `inference`, `runtime`, `tool-use`, `zo`.
- Every JavaScript binding is `pie-core-*` / `pie-instruct-*` / `pie-zo-*`; the
  only pie package that exists is `pie:inferlet`. For JS the PACKAGE is the
  sharper signal — bindings whose interface name happens to have survived the
  move (`model`, `session`, `chat`, ...) are just as unreachable as
  `inference`.

It is deliberately NOT in `ci.yml`: the breakage predates this work, and
red-lighting all of CI for it would be someone else's emergency. Gating the
publishes stops it from reaching users, which is the actual harm.

The port itself is its own project and should not be smuggled into this one.

### 10.5 D — per-stage PTIR containers  *(separate PR)*

Unchanged from §7 / §9.1. D4(a) hooks-as-methods and D4(b) per-stage containers
must land together; (a) alone would force the host to reassemble and
canonicalize containers, which is the thing D4 exists to avoid.

### 10.6 Loose ends

- `model::pass_kind()` returns `recurrent` only when `kv_page_size() == 0`,
  which no registered model satisfies. That arm is unreachable and therefore
  untested, and `forward-recurrent.wit` has no in-tree consumer. Left as is:
  the classification is correct and a pure-Mamba model would reach it — the
  gap is in the model zoo, not the code.
- ~~The `fold-buffered` no-logits claim~~ — SETTLED, and the doc was right.
  `qwen3_5_forward.cpp:1206` returns from the linear layer as soon as a fold
  boundary is set, BEFORE the output projection, so those layers contribute
  nothing to the residual stream and whatever reaches `lm_head` is missing
  them. `gdn-foldcommit` "passes" only because it never asserts on the value.
  The rule now lives on `rs-geometry.fold-len` in the WIT, where a guest will
  actually see it, rather than on a method that no longer exists.
- ~~§8.2's Metal RS flag-bit collapse~~ — FIXED.
  `batch_schedule.hpp` truthiness-tested `rs_slot_flags[r] != 0`, but
  `RS_FLAG_FOLD` (2) is also non-zero, so a FOLD row was read as a fresh
  sequence and `forward.cpp` then called `reset_state(slot)` on it — zeroing a
  live recurrent state. It now masks with RESET, as CUDA already did in both
  places it reads the flag. The constant is mirrored locally rather than
  included: `paged_batch_validation_test` compiles this header without the
  generated-ABI include directory.

### 10.7 First execution on real weights — three blockers, all silent

Everything above was written without ever running a linear model: the box had
no weights. Downloading Qwen3.5-0.8B and pointing the CUDA driver at a GPU
surfaced three defects in a row, none of which any unit test could have caught,
and all of which sat between the guest and the fold path.

**(a) No inferlet could load at all.** `34d38027f` bumped the package to
`pie:inferlet@0.3.0`; `process.rs` still looked up `pie:inferlet/run@0.2.0`.
wasmtime's export map is semver-exact, so every program failed with "No 'run'
interface found". The constant's comment already warned it must track
world.wit — a comment is not a mechanism, so it is now pinned by
`run_interface_version_matches_wit`, which parses the version out of the WIT.

**(b) The empty-buffer append was refused as a non-empty one.** `rs_plan_for`
read occupancy as `buffer_size() * page_size`, a page-granular upper bound. A
guest must reserve a page before buffering into it, so a brand-new empty buffer
claimed a full page and every speculative window died at submit. The host
already knew the exact figure — each buffering fire names its `(start, len)`
span — so the store now publishes it and keeps an exact `buffer_fill`.

**(c) Fold was unwritable.** An absent `readout` binding does not mean "sample
nothing"; the host synthesizes each lane's last row, and an empty readout
channel has no expressible shape. So the host gave every fold fire a sample row
it never requested, and the driver then refused the fire for requesting it
(§10.6). The default is now dropped once the plan is known to fold; an EXPLICIT
readout is still refused, loudly. Both fold-commit inferlets had been written
against the broken shape and had never run — each sampled `intrinsics::logits()`
in its commit fire while its own comment claimed "no logits".

**Result.** `mtp-native-verify` completes the full
buffer -> verify(`fold_len=0`) -> commit(`fold_len=accepted`) loop on an L40S
against Qwen3.5-0.8B: 14 windows, 22 tokens committed. The §11 model is now
executed rather than merely asserted. Acceptance quality is a separate question
(`mean_accept` is low, consistent with the known FLA commit-advance caveat) —
what is settled is that the shape runs.

**Toolchain note.** CUDA 13 needs an r580+ driver; an r550 host must build
against 12.x, which `swap_pool.cpp` had broken by calling `cudaMemcpyBatchAsync`
unconditionally (the signature differs across the 12/13 boundary). Now guarded,
with the per-copy fallback restored below 13.0.

**Still refused, still unimplemented:** the buffer READ path (§10.2). Nothing
here relaxes it — (b) only makes the classifier's input honest, so the position
that was always legal is now reachable.

---

## 10.8 Running the fold path end to end — the over-broad logits rule

§10.7 got `mtp-native-verify` through buffer -> verify -> commit. Building the
first GPU test for `gdn-foldcommit` (`bin/pie/tests/cuda_gdn_foldcommit.rs`)
immediately failed at frame step 0 with

    generated fused value shape exceeds u32

which was two faults wearing one message. `describe_generated_value` raised it
both for a shape whose product overflows u32 and for a dimension that resolved
to ZERO -- opposite causes, since a zero extent means a symbolic extent was
never bound and has nothing to do with size. Splitting the message named it at
once: symbolic extent #4, `PTIR_EXTENT_SAMPLED_ROWS`, resolving to zero.

The cause was `aa359f825`, from §10.7. It suppressed the synthesized read-out
for any plan that folds, justified by the rule *"a fold fire produces no usable
logits"*. That rule is about REPLAY, and folding is far wider than replay:

- `RsPlan::FoldBuffered` replays activations an EARLIER fire computed into the
  buffer slabs. It returns before the output projection, so it genuinely has no
  logits. This is the case the rule was written for.
- `RsPlan::Fold` advances the folded state over the fire's OWN new tokens,
  through the full backbone. **That is what a prefill on a linear model is.**
  Its logits are ordinary and required. And a pass with no RS working sets at
  all classifies as `Fold` too, so the suppression also silently removed the
  read-out from every attention-model fire.

So the "no logits" rule, applied to `Fold`, disabled generation on every
architecture at once. The driver had it right the whole time -- its
`rs_is_fold` is `mode == BufferFold`, nothing wider -- and the WIT text was
right too ("a fire that REPLAYS BUFFERED tokens"). Only the engine generalized
from the sentence rather than from the mechanism. The predicate now matches the
driver's.

This also accounts for `cuda_chat_completion_e2e`'s "fused value shape exceeds
u32", recorded in §10.7 as pre-existing and unrelated. It was neither: same
bug, reached through the attention path. That test now runs the model and fails
later and differently ("ptir: descriptor channel 0 not ready" during decode),
which IS unrelated to RS and remains open.

### What the new test pins

`cuda_gdn_foldcommit.rs` has two cases, and the split is the point:

- `one_chunk_folds_from_the_buffer` -- buffer 4, fold 2, abandon 2. Passes on
  real weights; the first execution of `gdn-foldcommit` ever. Its commit fire
  needed `epilogue(|| {})`: a pass with no stages has no PTIR program and
  registration fails outright, so an empty epilogue is the minimal well-formed
  program that samples nothing.
- `two_chunks_need_the_buffer_read_path` -- append a SECOND chunk onto the
  unfolded tail. The read path in its smallest honest form. It asserts the
  refusal names the BUFFER, and confirms the exact occupancy from `f1f498e0a`
  on hardware ("2 buffered token(s)", not a page). It is written to flip: when
  the read path lands it asserts the chain folded instead.

### Lesson, again

§10.7's lesson was "a warning comment is not a mechanism". This one is
narrower and sharper: **a rule stated in prose gets applied by its wording, not
by its reason.** "A fold produces no logits" was a true sentence about one
plan, generalized to a predicate over three. Both the driver and the WIT
carried the precise version; the imprecise restatement is what shipped. Where a
rule has a mechanical form -- here `mode == BufferFold` -- the code should name
that form and not a paraphrase of it.

#### 10.2.9 A commit is a row that carries no tokens — LANDED

The planner had three RS positions: `Fold`, `Buffer`, `Commit`. The first two
mix freely; the third does not, because a commit replays buffered activations
out of the slabs instead of computing them. It was selected by the condition
`fold_len <= buffered`.

That condition is not a statement of intent. It is an *incidental property* of
a commit that happens to hold — and it is also, exactly, the property of a
fire that folds BEHIND its own new tokens while writing them. One condition,
two opposite meanings, and the planner had no way to tell them apart. So the
shape the north star needs (`0 < n < b` with `t > 0`) was refused not by a
decision but by an ambiguity.

**A row that spans no tokens is the unambiguous form.** "I have nothing to
compute; only move the boundary", said directly. It adds no flag, no field and
no WIT surface: the guest simply declares an empty token span.

##### Why it has to live in the geometry, not in a channel

The obvious encoding — an empty token channel — does not exist. `Shape::new`
(`compiler/ir/src/types.rs`) refuses a `0` dim: the IR has no zero-sized
tensor, and adding one is a change to the whole type system for the sake of
one degenerate row. `bind_window` in `mtp-native-verify` had already hit this
for `readout` and worked around it by *omitting the port entirely*.

The emptiness therefore lives where it already belongs: in `qo_indptr`. Every
per-token channel carries at least one element, and the CSR says how many of
them this fire actually spans. Two consequences, both now enforced host-side
in `pipeline/fire/geometry.rs`:

  * **The token CSR is the truth.** `token_ids` and `position_ids` are
    truncated to `qo_indptr.last()`, and refused if they are shorter. The ABI
    requires `qo_indptr[rows] == token_ids.len` exactly (`validate_csr`), so
    the unreferenced tail must be dropped before the wire, not on it.
  * **An empty lane samples nothing.** The defaulted readout was
    `lane[1].saturating_sub(1)` per lane, which for `[0, 0)` yields row 0 —
    outside the lane — and the sampling CSR then failed to partition. Empty
    lanes are now filtered out of the default.

##### What the classifier became

```
t == 0            -> Commit   (and n == 0 is refused: the fire would do nothing)
n == 0            -> Buffer
n == b + t, b == 0-> Fold
otherwise         -> Buffer
```

`n <= b` no longer appears. Every row that computes anything is a `Buffer`
row with a cut at `n`, and the driver's 2R-segment split (§10.2.8) already
handles a cut anywhere in `(0, b + t)` — including behind the new tokens. So
**fold-behind-while-writing-ahead came for free**; it was only ever blocked by
the classifier.

##### The blast radius is the point

Every existing commit passed a *placeholder token* to satisfy the old shape,
and under the new rule that token is no longer ignored — it is a real token,
buffered and written to KV. Three GPU tests changed their answer the moment
the classifier did, which is precisely the evidence that the old encoding was
carrying meaning it never declared. All commit sites (`gdn-foldcommit`'s four
arms, `mtp-native-verify`'s `build_commit`) now use the empty row, and
`build_commit` lost its "re-run the full window" apology along with it.

Verified: `cuda_gdn_foldcommit` 8/8 one process each, including the new
`a_commit_may_carry_no_tokens_of_its_own` (an empty-row fold vs no fold at
all — folding is a no-op, so they must agree) and
`a_fire_may_fold_behind_the_tokens_it_is_writing` (one fire vs two);
`mtp_logits_value_verify` and `a_device_resident_fold_length_decodes_identically`
decode identically to before. Metal refuses a zero-token row.

##### What still stands between this and one fire per window

Everything the *planner* needed is now in place. What remains is not a
programming-model question but a device-geometry one, and it is worth stating
precisely so it is not rediscovered:

A window's buffer holds `[accepted | rejected]`. Folding `clen` in the NEXT
window's fire leaves the REJECTED tail sitting between the boundary and the
new tokens, and the recurrence would replay it. Removing it needs the buffer
READ LENGTH to be device-resolved from the same number as the fold length —
`clen` — since the host may not know it. That is one more substitution in
`batch_compose::append_rs`, in the same place `PIE_RS_FLAG_FOLD_LEN_DEVICE`
already substitutes, and it is the last piece.

#### 10.2.10 `discard` is the missing dual of `fold` — THE NORTH STAR, LANDED

One fire per speculative window. The thing §11.4 wanted and could not have in
the form it asked for.

##### The buffer only had one movable end

```
folded[0, F)              buffer[F, F+B)
        `-- fold(n): F moves RIGHT.  Commit. Irreversible.
                              discard(n): F+B moves LEFT.  Abandon. Free.
```

The boundary model gave the guest one of these. The other did not exist: the
only way to get rid of buffered tokens was `free-buffer`, which empties the
buffer WHOLESALE.

That single asymmetry is the entire reason a speculative decode needed two
fires. A verify fire buffers `k+1` tokens, of which a prefix is accepted.
Dropping the rejected tail meant `free-buffer`, which would take the accepted
prefix with it — so the prefix had to be folded away FIRST, and it could not
be folded by the verify fire itself, because its length is not known until
that fire's own logits come back. Hence a second fire whose only job was to
move the boundary so the buffer could be emptied.

The commit fire was never a goal. It was a workaround for a missing verb.

##### Why not in `rs-geometry`

`fold-len` lives in the geometry because it changes what the DEVICE does: it
is where the recurrent-state snapshot lands. `discard` changes nothing on the
device at all — the slots it releases are simply overwritten by the next
append. Putting a device-side no-op in the geometry would make the geometry
mean two things, "what to compute" and "what to forget".

The WIT already records the precedent, in the comment that buried
`start-token`: *"letting the guest state a runtime-owned value is what made
multi-chunk buffering expressible and wrong."* The buffer's live extent is
runtime-owned in exactly that sense. `discard-buffered` does not STATE it; it
SHRINKS it, which is what a resource method is for. It is also the
token-granular sibling of `free-buffer`, and the two now divide cleanly:
**`free-buffer` releases CAPACITY, `discard-buffered` releases CONTENT.**

There is a phase argument too. `fold` is something a fire does — the boundary
lands where the device writes state. `discard` must happen before the fire is
even planned, since the new tokens overwrite the discarded slots. A record
field has no way to carry "this one applies before everything else here".

##### The loop

```
read window k's outputs      (the loop already does this -- it needs the drafts)
rs.discard_buffered(k+1 - clen_k)
fire(window k+1 tokens, fold_len = clen_k)      <- ONE fire
```

`b = clen_k`, `t = k+1`, `n = clen_k`: exactly the fold-behind-while-writing-
ahead shape §10.2.9 opened. No ABI change, no new descriptor, no
device-resolved read length.

##### The device path deliberately does NOT fuse

This inverts the assumption §11.4 was written under. A device-resident
`fold-len` is never read back, so the store holds only an UPPER BOUND on the
live buffer and refuses to plan a replay against it — and a fused fire must
replay the very prefix it is folding. So:

  * **device-residency** buys back-to-back enqueue WITHOUT a host read;
  * **reading back** — which this loop does anyway, for the next drafts —
    buys the fusion.

They are alternatives, not a ladder. `mtp-native-verify` now runs both:
`fires=14` on the host path against `fires=28` on the device path, over
`steps=14` windows, decoding the identical stream.

##### Verified

`mtp_logits_value_verify` (host, fused) and
`a_device_resident_fold_length_decodes_identically` (device, two-fire) produce
identical token streams; the harness asserts `steps == fires` on the host path,
because a regression to two fires would not change a single token. Store unit
test `discarding_buffered_tokens_releases_content_but_not_capacity`;
`cuda_gdn_foldcommit` 8/8; engine lib 376 pass / 2 known timing;
`--all-targets` clean. Metal needs no refusal: `discard` never reaches a
driver.

#### 10.2.11 Review of the north-star work — three real defects

A full review of §10.2.8–§10.2.10 found three things that were wrong rather
than merely untidy. All three share a shape: a check or an obligation that was
correct only because some OTHER layer happened to make it so.

**A commit shipped an unclamped fold length.** `fold-len` is documented as
CLAMPED to `B+T`, so "fold everything" is written `u32::MAX`. Under the old
`fold_len <= buffered` classifier a commit was only ever SELECTED when its
length already fit, so passing the raw guest value through was invisibly safe.
`t == 0 -> Commit` removed that coupling: a commit may now carry any length,
and `u32::MAX` reached `validate_fold` and was refused. The planner already
computed the clamped `fold_tokens`; the commit arm simply did not use it.
`empty_commit` gained an arm that folds `u32::MAX` through an empty row.

**The deferred fold and the in-flight hold leaked on four error paths.**
`publish_batch` returns a `#[must_use]` `RsPendingFolds` plus a receipt, and
the wire arrays are built between the publish and `commit_folds`. That window
has four fallible steps, one of them guest-reachable. On any of them the
pending folds were dropped — leaving the boundary where nothing expects it —
and `out.txn` was dropped without `settle`, so `in_flight` never decremented,
and `retire_idle` gates ALL pool retirement on `in_flight == 0`: one such
error wedged RS slot recycling for the life of the process. `#[must_use]` does
not catch this, because the value is bound to a local. The window is now a
closure whose result is inspected only AFTER both obligations are discharged.

**`validate_fold` measured page capacity, not live tokens.** A two-page buffer
holding three live tokens accepted a fold of six, gathering slab tokens that
were never written into the recurrent state — silent corruption, not a visible
failure. Nothing hit it because the planner pre-clamps, which is exactly what
makes it a trap for the next caller. The bound is now the LIVE extent, and the
intent says what that means: a REPLAY (a commit) may reach `buffer_fill`; a
WRITE may reach `start + len`, because the fire's own new tokens are part of
the extended space it folds through. Capacity is still checked on top, because
the gather is physical.

Also fixed, all smaller: `discard_buffered` did not rebase `buffer_head` when
it emptied the buffer (`advance_fold` and `free_buffer` both do, and without it
a repeatedly-drained buffer walks its head up until a fresh window cannot hold
its own tokens); the zero-token truncation was an unconditional `truncate`,
which quietly absorbed the ABI's exact-length invariant for EVERY fire instead
of only the padded one; three Metal refusals were wrapped in a shape test, so
an unexpected array shape SKIPPED the safety check rather than failing;
`PIE_RS_SPLIT_TRACE` gated on `layer_idx == 0`, which on a hybrid stack may be
an attention layer the linear body never runs for; `RsPendingFolds` carried a
positional `(ws, tokens, is_bound)` tuple; and `Qwen3_5RsFoldSplit::qo_h` was
assigned and never read.

##### Verified

`cuda_gdn_foldcommit` 8/8, `mtp_logits_value_verify` (`steps=13 fires=13`),
`a_device_resident_fold_length_decodes_identically` (`steps=13 fires=26`),
engine lib 377 pass / 2 known timing, `cargo check --workspace --all-targets`
clean, Metal syntax-checks.

#### 10.2.12 What a recurrent fire costs, and why — measured

Benchmarking against vLLM 0.25.1 on one L40S turned up one hard failure and
one large, precisely-located cost. They have the SAME root.

**The failure.** Every hybrid request died at the default frame size with a
cascade — `descriptor channel 0 not ready` from the driver, then a poisoned
guest channel — while `PIE_FRAME_SIZE=1` passed. The cause is a collision
between two layers that had each moved:

* `validate_frame` dropped its "an RS pass owns its frame" rule once RS
  mappings began publishing at prepare (§10.2.x), so `live_slots()` returned
  k for a linear model and a run-ahead lane happily filled a frame with
  chained decodes.
* The CUDA driver never gained the matching ability. `FramePrepare` runs
  EVERY step's host work at frame entry, before any of the frame reaches the
  stream. A decode fire normally escapes that by taking the device-composed
  template, which resolves its ports at kernel time — and
  `try_device_composed_template` (`driver/cuda/src/pipeline/dispatch.cu`)
  bails on any non-empty `rs_slot_ids`. So an RS fire resolves its ports on
  the HOST at frame entry and demands a cell that an earlier slot of the same
  frame has not produced.

Fixed on both sides of the boundary. `live_slots()` now returns 1 for a
recurrent model, so a well-behaved lane never builds that frame; and
`validate_frame` refuses it by name, with the slot pair and the reason, so a
lane that builds it anyway gets an actionable error at submit instead of a
device cascade. Regression-tested by the `coframe` arm of
`bin/pie/tests/cuda_rs_buffer_bench.rs`.

**The cost.** `PIE_STEP_PROFILE=1`, 1-wide decode:

| phase | attention (Qwen3-0.6B) | hybrid (Qwen3.5-0.8B) |
|---|---|---|
| prepare | 16.2 us | **3131.2 us** |
| enqueue | 471.5 us | 701.5 us |
| settle | 56.6 us | 64.1 us |

The 193x is the same host descriptor readback. It does not merely copy — it
waits, because the token it reads is produced by the previous fire on the
device. A recurrent decode lane is therefore serialized through the host on
every single step.

Two consequences worth stating plainly:

* **The RS buffer's cost today is not its write.** Binding `fold-len` as a
  channel takes a fire out of the decode-envelope class entirely —
  `classify_decode_envelope` has no case for ANY RS port, so it falls back to
  host-evaluated geometry, which then refuses a device-carried token
  (`EmbedTokens is not host-derivable`). A buffered decode loop cannot be
  device-carried at all. That is a real programmability gap: a speculative
  decoder's drafts are device-produced by construction.
* **Both of these dissolve if `try_device_composed_template` learns to carry
  RS rows.** The RS wire arrays (`rs_slot_ids`, `rs_fold_lens`,
  `rs_buffer_slot_ids`) are host-known — they come from the engine's RS
  store, not from device channels — so the template's refusal looks like a
  gap in the compose kernels rather than an intrinsic one. That is the single
  highest-value item left on the linear track.

**Measured against vLLM 0.25.1** (one L40S, driver 550, shared
`benches/common.py` metrics, `VLLM_USE_FLASHINFER_SAMPLER=0`):

Steady state, after merging `origin/dev` (which brought a chunked argmax
accumulator and two vec2 alignment fixes). Throughput is the mean of three
runs -- pie's first run after a cold start is an outlier, vLLM's spread is
under 2%:

| case | pie | vLLM | delta |
|---|---|---|---|
| attention latency 16x128 | 437.72 tok/s | 376.43 | **+16.3%** |
| attention tput 256x128 | 27683 | 27484 | **+0.7%** |
| hybrid latency 16x128 | 266.44 | 318.14 | **-16.2%** |
| hybrid tput 256x128 | 13539 | 11408 | **+18.7%** |

pie leads in three of the four. Two things moved it there: the frame-size fix
above took hybrid throughput from 11717.80 (handicapped at
`PIE_FRAME_SIZE=1`) to ~13539, and dev's argmax/alignment work took attention
throughput from 24795 to ~27683, closing what had been a -8.9% gap.

The one remaining gap is hybrid latency, and it is NOT host overhead. The
arithmetic: pie's measured forward time for a 1-token hybrid decode step is
3.41 ms, against vLLM's 3.14 ms END-TO-END. Host work adds only the
non-overlapped remainder -- prepare's 3131 us is spent WAITING on that same
forward, so total step time (3.79 ms) is forward time plus about 0.38 ms of
enqueue and settle. Even with zero host overhead pie would sit at 3.41 ms,
i.e. ~293 tok/s against 318. The gap is GDN kernel work, not plumbing.

**On relaxing the template's RS guard.** Investigated, and it looks
conservative rather than intrinsic:

* The compose kernels ALREADY know about RS -- both the envelope and
  fixed-decode compose paths carry an `output.rs_slot_ids` and write `-1` into
  it for inactive (padded) rows (`dispatch.cu:752-754`, `1119-1121`).
* The RS wire arrays are consumed by the MODEL FORWARD
  (`driver/cuda/src/model/qwen3_5/qwen3_5_forward.cpp`), reached through
  `ComposedBatch` (`driver/cuda/src/batch/batch_compose.hpp`), not by the PTIR
  compose kernels the template replaces. They are host-known: the engine's
  `RsStore` produces them.
* `git log -S "rs_slot_ids.empty"` turns up only the original PTIR fusion
  commit and two `update`s -- no commit that revisits the guard deliberately.

What is genuinely RS-specific is `rs_launch_requires_readiness_settlement`
(`driver/cuda/src/batch/rs_metadata.hpp:20`), which forces two D2H readiness
settlements per RS launch because "stateful model launches cannot discover a
ticket miss after the recurrent-state kernels have already mutated their
slots". That is sound and costs about 230 us of the enqueue phase -- it is not
the 3131 us.

So relaxing the guard is worth doing, but for PROGRAMMABILITY first (it is
what would let a buffered RS decode stay device-carried, which a speculative
decoder needs because its drafts are device-produced) and for latency only
second, where the ceiling is about 10%.


#### 10.2.13 Where the hybrid latency gap actually was — a kernel nobody could reach

The conclusion above ("it is GDN kernel work") was right, but the kernel in
question was not slow for any interesting reason. It was slow because the fast
version of it was unreachable.

`nsys` on a 1-token hybrid decode put `recurrent_step_batched_kernel` at 20.5%
of all GPU time: 3348 launches, **41.6 us each**, standard deviation under 1 us.
At 18 linear layers that is 749 us per decode step. The state is only 512 KB
per layer (16 heads x 128 x 128, bf16), so two reads and two writes of it is
2 MB, which an L40S moves in 2.4 us. The kernel was 17x off, and the reason is
visible in its launch: `dim3 grid(R, V_h)` with R=1 is **16 blocks on a
142-SM GPU**.

A tuned replacement already existed and was already the default --
`recurrent_step_batched_gqa_smem_kernel`, which stages the state tile into
shared memory once and reads it from there in both phases. It was reachable
only through the GQA launcher, and the dense forward gated that launcher on

```
const bool use_decode_gqa_recurrent = ... && V_h != K_h && V_h % K_h == 0;
```

`V_h != K_h` was not expressing a real constraint. Repeat-1 is just the
degenerate grouped case: the GQA kernel computes `h_k = h/1 = h` and indexes q
at `(r*K_h + h)*K_d`, which is the non-GQA kernel's `(r*V_h + h)*K_d`, and the
two call sites hand it the same pointer, because `q_recur_full` IS `la.q_pre`
when `V_h == K_h`. So a model with equal linear key and value head counts --
Qwen3.5-0.8B has 16 and 16 -- silently took the slow path. Dropping the
clause: **41.6 us -> 7.57 us per layer**, 20.5% -> 4.5% of GPU time.

**The trap.** That change alone made `cuda_mtp_stage1` fail its golden, and
the golden is not a self-captured regression baseline -- it is the HF
transformers trajectory for the same prompt. The legacy kernel reproduced all
24 tokens; the SMEM kernel diverged at the **second** one. Both kernels are
correct: a new CPU-reference parity test
(`driver/cuda/tests/gdn_recurrent_step_parity.cu`, 13 shapes across R and
repeat) put the SMEM kernel within 3e-6 of an fp32 reference where the legacy
kernel sat at 1.7e-3. The SMEM kernel was the *more* accurate of the two and
still produced different text.

The difference was one rounding point. The legacy kernel stores `state*g` to
HBM in its first phase and reloads it in its second, so its phase-2 base is
bf16-rounded; the SMEM kernel keeps the value in a register and stays in fp32.
Under argmax that is enough to pick a different token as soon as two logits
are close -- and at the second decoded token there is no accumulated drift to
blame, just a near-tie. So the fix is to round in the same place:

```
const float sg = __bfloat162float(__float2bfloat16(
    __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h));
```

**Kernel selection is an implementation detail and must not be observable in
model output.** The parity test now holds every kernel to the legacy rounding
so this class of difference cannot land silently again -- which matters beyond
this bug, because the SMEM kernel was already the production default for every
grouped-head Qwen3.5 model, and nothing was checking it against the dense one.

Result on one L40S, against vLLM 0.25.1:

| case | before | after | vLLM | delta |
|---|---|---|---|---|
| hybrid latency 16x128 | 266.44 | **318.95** | 318.14 | +0.3% |
| hybrid tput 256x128 | 13539 | **15127** | 11408 | +32.6% |

Latency was -16.2% and is now parity; throughput went from +18.7% to +32.6%.
The rounding fix costs nothing measurable. Attention is untouched (437.81 vs
437.72 tok/s). The forward-time arithmetic in §10.2.12 still stands -- pie's
remaining non-overlapped host cost is ~0.38 ms -- but the 3.41 ms forward it
was measured against is now roughly 0.61 ms shorter, which is the whole gap.


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
