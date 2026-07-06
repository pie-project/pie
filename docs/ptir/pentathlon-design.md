<!-- Capstone stress-test design (echo). Composes overview §6.1 + §6.2 + §6.4
     into one scheme; authoritative semantics remain overview.md. -->

# Pentathlon+1 — MCTS-guided, grammar-constrained, MTP-speculative,
# contrastive beam expansion under Quest attention

**Status:** Design + gap analysis (the stress-test deliverable). Implementation
follows §6.2-e2e + real-MTP landing.
**Composes:** Quest (overview §6.1) · MCTS (§6.4) · beam (§6.2) · constrained
decoding (§6.1/§3) · MTP speculation (§6.1) · **contrastive decoding (§6.3
design note)** — all six in one inferlet, designed for high-throughput fleets
under the quorum scheduler (§7.2).
**Verdict up front:** the six-way composition still needs **zero new core ops
and zero new first-party intrinsics** — the op set closes. Contrastive is the
first technique to expose gaps BENEATH the op set (§5 G6): the amateur's
logits bind as an ordinary channel (option (a) — no second-model intrinsic,
exactly where §6.3's design note parked it), but realizing the efficient form
needs (i) **multi-model runtime execution** (spec'd as direction) and (ii) an
**extern-channel binding in the PTIR container** (the registration surface
cannot yet express §1's "pairs may span pipelines"). A host-fed interim works
today. Everything else: host-clock placement or perf probes, plus the §5 G1
in-graph-DFA finding.

---

## 1. The scheme

Task shape: generate grammar-valid output (e.g. JSON conforming to a schema,
or tool-call DSL) from a long context, optimizing a terminal objective scored
by `value_head` — better than greedy/beam alone can, at fleet throughput.

- **MCTS** is the outer search — a host *process* (§6.4: unbounded,
  data-dependent control flow never enters a trace). Nodes = sequence
  prefixes; the KV tree IS the page-aliasing structure (§5.2).
- **Beam** is the expansion rule: a selected leaf expands to its top-B
  grammar-legal children in one forward (per-node beam; priors from the
  masked distribution).
- **Constrained**: every sampled/expanded token passes the grammar mask —
  per-position masks under speculation, exactly the §6.1 discipline.
- **Speculative (MTP)**: node evaluation = a depth-D rollout decoded with
  the §6.1 MTP window (draft K, match-verify, lossless commits) — the
  rollout is D/(E[n_acc]+1) passes instead of D.
- **Quest**: every forward (expand + rollout) selects top-`budget` KV pages
  per layer via `envelope_dot` → `attn_page_mask` — long-context efficiency
  orthogonal to all of the above.
- **Contrastive**: the pick rule scores
  `cd = log_softmax(expert) − λ·log_softmax(amateur)` under the plausibility
  constraint (keep token iff `log p_expert ≥ log α + max log p_expert`), then
  the grammar mask, then top-k/argmax. The amateur's logits arrive **through a
  channel** (§2.4) — never a new intrinsic.

One MCTS iteration, batched R-wide (virtual loss = run-ahead, §6.4):

```
select R leaves (PUCT + virtual loss)          [host]
  ├─ EXPAND pass per leaf  → top-B children (tokens, priors) + leaf value
  └─ ROLLOUT pipeline per frontier child → D tokens via MTP+grammar
       → terminal value_head
backprop values, undo virtual loss             [host]
alloc/free pages exactly (host owns node→slot) [host]
```

## 2. The traced programs (two identities + prefill)

Everything below is core-appendix ops + registered intrinsics. `B` = beam
width per expansion, `K` = MTP draft width, `P_MAX` = trace-known page cap,
`V` = vocab. Channels per instance; geometry host-fed (§6.4 style — the tree
owns the slot map, so reclamation is *exact*).

### 2.1 `expand` — one lane, one token read, B children out

```rust
// channels: pages [P_MAX] u32 (host-fed, node's row), klen [1], kvm, pos,
//           mask [V] bool (host-fed: node's grammar state), budget [1] u32,
//           prio [B] f32, pids [B] i32, val [1] f32   (host-read)
fwd.attn_working_set(&ws, &pages, &klen, ..);   // node's path (aliasing = tree)
fwd.attn_mask(&kvm);
fwd.on_attn_proj(|| {                            // ── QUEST (per layer)
    let s = intrinsics::kernel::envelope_dot(intrinsics::query());
    intrinsics::kernel::attn_page_mask(pivot_threshold(s, rank_le(budget.read())));
});
fwd.epilogue(|| {
    let lp = log_softmax(mask_apply(intrinsics::logits(), mask.take()));
    let (pr, ids) = top_k(lp, B);                // ── BEAM (grammar-legal by
    prio.put(pr); pids.put(ids);                 //     construction: -inf never wins)
    val.put(intrinsics::value_head());           // ── MCTS leaf value
});
```

### 2.2 `rollout` — the §6.1 program + a value tap

Verbatim overview §6.1 (MTP window `[committed, d1..dK]`, per-position mask
`[K+1, V]`, match-verify `eq→cumprod→select`, `-1` sentinel, cursor advance,
Quest tap identical to 2.1) **plus one line** in the epilogue:

```rust
    val.put(intrinsics::value_head());           // [K+1] or last-row value
```

The host reads `val` every step and *uses* only the step where the rollout
reaches depth D (or terminal EOS) — see gap G3 for why this is a put-per-step
rather than a conditional put. Rollout geometry: the child aliases the node's
pages (freeze discipline, §5.2/§6.2); its D·(≤K+1) appended tokens land in
child-private tail slots the host frees exactly after backprop.

### 2.3 Contrastive pick rule (drop-in for 2.1/2.2 epilogues)

Pure appendix ops. `am` is the amateur-logits channel (§2.4); λ, log α are
`Tensor::constant`s.

```rust
let lse = log_softmax(intrinsics::logits());              // [rows, V] expert
let lsa = log_softmax(am.take());                         // [rows, V] amateur
let cd  = sub(lse, mul(LAMBDA, lsa));                     // contrastive score
let lse_g = select(gmask, lse, NEG_INF);                  // grammar FIRST
let plaus = ge(lse_g, add(broadcast(reduce_max(lse_g), SH), LOG_ALPHA));
let scored = select(plaus, cd, NEG_INF);                  // α within legal set
// expand: top_k(scored, B)   ·   rollout: picked = reduce_argmax(scored)
```

**Composition-order pin (found BY the `pentathlon_iter` golden):** the
plausibility max is taken over the **grammar-constrained** expert
distribution, never the raw one — α against the unmasked max annihilates
every legal token whenever the grammar masks the expert's peak (leaf B in the
golden scored uniformly −inf in the first cut). Order: grammar → plausibility
→ contrastive. The `-inf` fold makes the grammar∧plausibility conjunction one
`select` + one `ge`.

In the rollout the contrastive rule REPLACES the pick over the masked target
distribution; the match-verify tail (`eq→cumprod→select`, golden
`mtp_verify_tail`) is unchanged — acceptance compares drafts against whatever
`picked` is, so speculation stays lossless w.r.t. the contrastive-argmax
choice. The amateur must score the same K+1 window rows, i.e. its pipeline
runs the same speculative geometry (`am` is `[K+1, V]` in the rollout).

### 2.4 Where the amateur's logits come from — the stress-test question

Three bindings, in order of preference; NONE adds an op or intrinsic:

- **(a-device) cross-pipeline channel (the target form).** A second pipeline
  runs the amateur model; `am`'s producer endpoint is the amateur's epilogue
  (`am.put(intrinsics::logits())`), consumer is the expert's — §1 explicitly
  allows SPSC pairs to span pipelines ("SPSC constrains endpoints, not
  clocks"; §6.3's design note names exactly this). Zero host round-trip.
- **(a-host) host-fed channel (works TODAY, the interim).** The host runs the
  amateur (second request / smaller model) and feeds `am` mask-style. Fully
  expressible now; costs a `[rows, V]` f32 host edge per step (§5 G6 probe).
  Bandwidth mitigation with existing ops: feed only the amateur's top-M
  `(ids, logprobs)` (M trace-known) and reconstruct in-graph —
  `scatter_set(broadcast(AM_FLOOR, [V]), am_ids.take(), am_lp.take())` —
  M·8 B/row instead of V·4 B/row; the plausibility constraint makes the
  floor exact for CD's argmax on the plausible set.
- **(b) a second-model-logits intrinsic — REJECTED.** An in-program
  `intrinsics::amateur_logits()` would weld two models into one trace: pass
  identity is the tuple of ONE forward's stage traces (§5.3), capacity
  pricing and the fire rule assume one trunk per pass, and D2 would be
  violated (which-amateur becomes trace identity, not data). The channel form
  keeps the amateur swappable per-instance. This is the precise answer: the
  model binds a second model's logits as DATA, never as an intrinsic.

### 2.5 Identities and batching

Two epilogue traces ⇒ **two program-set identities** (C3) fleet-wide, both
shared by every request (per-instance data: seeds, budgets, masks, geometry —
D2 keeps identity clean). Under the quorum rule (§7.2): expand instances and
rollout instances co-batch per identity; a pipeline blocked on host work
(select/backprop is value-dependent) is absent from the denominator, not
awaited — MCTS's think-time rides the escape clause by design.

## 3. Host orchestration

```
tree: nodes { prefix, pages row, grammar_state, N, W, P, children }
matchers: grammar matcher pool — one live matcher per in-flight rollout,
          forked from the node's state (CFG: host-cloned pushdown state)
loop:
  leaves = PUCT_select(R, virtual_loss)                     // stats trail ≤1 batch
  for l in leaves: feed(expand_l geometry+mask); submit
  for c in frontier_children: submit rollout step; feed mask_{t+1}
      from matcher.speculative_masks(published drafts)      // §6.1 host walk
  harvest: prio/pids/val (expand), out/draft_out/val (rollouts)
  expand tree, backprop, undo virtual loss
  ws.alloc headroom (in-flight safe); ws.free(pruned + retired rollout tails)
  every GC_EVERY: compact(live_runs) if fork-waste high     // §6.2 tier-2
```

Throughput probes (the "measured under high throughput" half): dummy-run rate
(late masks), quorum-escape rate (MCTS absentees), mask bytes/step host→device,
accepted-tokens/s vs plain §6.1, pages freed/iter vs RCU grace-period depth,
per-identity batch density.

## 4. SDK surface (delta's `sdk/rust/ptir`)

No new API. The author writes 2.1/2.2 with existing builders: `Channel`,
`ForwardPass::{attn_working_set, attn_mask, on_attn_proj, epilogue}`,
`intrinsics::{logits, value_head, mtp_logits, query, kernel::{envelope_dot,
attn_page_mask}}`, free ops incl. `top_k`/`log_softmax`/`mask_apply` — plus
the Stage-2 `[K, vocab]` `mtp_logits` binding (`ab8ec2f1`). The MCTS/beam
control loop is ordinary host code (wasm), per §6.4. This is itself a
finding: **the composition is expressible without touching the SDK.**

## 5. Gap analysis — the stress-test result

### What composes cleanly (with the reason it composes)

| pair | why |
|---|---|
| spec × grammar | acceptance is a value, masks are data (§6.1; golden `mtp_verify_tail`) |
| quest × everything | a per-layer sink + peeked budget adds no edge to any decode chain (§6.1) |
| beam × grammar | `top_k(log_softmax(mask_apply(..)))` — masked -inf never wins; pure appendix ops (golden `matrix_select_mask`) |
| mcts × beam/spec | the tree is host process + host-owned geometry (§6.4); programs never see the search |
| spec × mcts memory | K+1 provisional writes per rollout step + cursor advance are per-instance channel state; reject tails are overwritten, dead rollouts freed exactly |
| the fleet | two identities (+1 amateur trace under (a-device)), batch-by-program, quorum denominator excludes host-blocked pipelines (§7.2) |
| contrastive × spec | the pick rule is upstream of verify; match-verify is pick-agnostic (golden `mtp_verify_tail` unchanged) |
| contrastive × grammar/beam | one `and` composes plausibility ∧ grammar before top-k — pure appendix ops |

### G1 — grammar-mask production is the host-clock hot spot (and a NEW finding)

Under speculation each rollout needs `[K+1, V]` fresh masks per step —
produced by a per-rollout matcher on the host. At R rollouts that is
R·(K+1)·V/8 bytes/step host→device (packed) and R matcher walks on the
critical mask edge (the one host-coupled edge, §3). **Finding:** for
*regular* grammars the walk is expressible **in-graph today with existing
ops** — DFA next-state and allow tables as large `Tensor::constant`s
(resident immutable buffers, §1), state as a `[1]` channel,
`row = gather(allow, state)`, `state' = gather(next, state*V + tok)`, and the
K+1 speculative rows are K unrolled gathers along the draft tokens
(trace-known K). No new op — just the large-constant lowering path exercised.
CFG/pushdown grammars genuinely cannot move in-graph (unbounded stack =
process, not program) — that boundary is correctly drawn and stays host-side.
**Recommendation:** implement the in-graph DFA walk as the follow-up
demonstrator; it removes the mask edge entirely for regular grammars and
turns G1 from a scaling wall into a host-CFG-only cost.

### G2 — no cross-instance reduction (structural, intentional)

Global beam ("best B children across ALL expanded leaves") and any
tree-global argmax need a reduction ACROSS instances. The model has no such
op — a batch is independent instances by construction (identity ws-independent,
reductions row-local §7.3). The composition therefore pays one host round-trip
per MCTS iteration for global decisions. **Verdict:** correct boundary, keep
it; virtual-loss run-ahead already hides the latency (§6.4). Recorded so
nobody tries to "fix" it with a cross-instance channel — that would break
batch-by-program identity.

### G3 — no conditional channel effects (minor)

"Put `val` only at rollout depth D" is not expressible: a `put` is
unconditional; data-dependent choice is `select` on *values*, and §7.1's
register rule (double-put last-wins) assumes effects always land. Workarounds
are cheap — put every step (a `[1]` f32; host ignores non-terminal reads) or
a second traced variant for the terminal step (bucket churn). **Verdict:**
accept put-every-step; record `put_if` as a *rejected* extension unless a
composition surfaces where the effect payload is large (then revisit at spec
level, since predicated commit already exists per-pass in §7.1 — a per-op
predicate is a different, heavier contract).

### G4 — allocator pressure from rollout churn (perf probe, not semantics)

Rollout KV is transient: R·D·(K+1) provisional tokens per iteration allocate
and free within ~2 iterations. Exact free composes (§6.4), but the arena's
RCU grace period (§5.2) becomes the hot path — freed pages recycle only after
in-flight passes retire, so peak page residency ≈ live tree + 2 iterations of
rollout tails. **Probe:** pages-freed/iter vs grace-period depth; if headroom
`alloc` stalls, the fix is sizing (bigger arena / smaller R), never a new op.

### G6 — contrastive names the first gaps BENEATH the op set

The op-set verdict survives six techniques: contrastive is `sub`/`mul`/
`log_softmax`/`reduce_max`/`ge`/`and`/`select` + a channel. But the
**efficient** binding (a-device, §2.4) needs two things the runtime/container
do not yet have:

1. **Multi-model runtime execution** — two models (expert + amateur) driven
   by one inferlet, each its own pipeline. The overview records this as
   direction (§6.3 note, §6.5's cross-pipeline weight feed); channel
   SEMANTICS already cover it. Note a same-model amateur needs no multi-model
   at all: a context-truncated or different-working-set amateur is just a
   second instance of the SAME model — available as soon as cross-instance
   channels are.
2. **Extern-channel binding in the PTIR container (P0 gap — mine).** The
   container declares channels per-trace with `host_role ∈ {none, writer,
   reader}`; there is NO way to declare "this channel's other endpoint is
   another instance." §1/T2 permit it; the registration surface can't express
   it. Fix is append-only: a fourth host-role-like variant `extern(name)` (or
   an imports/exports table) + instantiation-time pairing, validator checks
   SPSC across the pair. I own this and will land it when multi-model
   scheduling work starts — it is a container v1.1 addition, not an op change.

Until then **(a-host) is the shipping form** — correct today, with the top-M
scatter mitigation bounding the edge. Probes: amateur-edge bytes/step (full-V
vs top-M), expert/amateur step-phase skew (two clocks, one SPSC channel —
back-pressure is the coupling), dummy-run rate on the `am` edge vs the mask
edge.

### G5 — everything else already existed

`value_head` (model-gated), `mtp_logits [K,V]` (Stage 2, `ab8ec2f1`),
`envelope_dot`/`attn_page_mask` (second-party, bind-time), per-position bool
masks (ruling B, 1-byte), `top_k` (immediate k) — and contrastive's whole
vector algebra (`sub`, scalar `mul`, `log_softmax`, `reduce_max`, `ge`,
`and`, `select`, `scatter_set` for the top-M reconstruction) — all present in
the P0 registry and golden-anchored. The six-way composition adds **no
entry** to the op table, which was the point of D5's closed core: if this
scheme fits, extension pressure is genuinely second-party (or, per G6,
runtime/container plumbing — never the IR).

## 6. Implementation plan (post-§6.2/MTP landing)

1. Mock-first: compose 2.1+2.2+2.3 in the tier-0 interpreter harness; golden
   `pentathlon_iter` (one MCTS iteration, R=2, B=2, K=3, host-fed amateur —
   a-host) — **DONE** (`tests/golden-ptir/pentathlon_iter.txt`): all six
   techniques fire with designed, checkable decisions (contrastive demotion
   in the beam, grammar-forced accept boundary, contrastive-broken draft
   chain, per-layer quest sinks, value taps). It caught the §2.3 order pin.
2. In-graph DFA demonstrator (G1) — **DONE**
   (`tests/golden-ptir/dfa_ingraph.txt`): allow/next tables in seeded
   read-only device channels, state a [1] ping-pong, mask row =
   `gather(allow, state)`, walk = `gather(next, state·V + picked)`. The
   readiness table shows ZERO host-writer channels — the grammar edge is
   deleted; three steps force 2→3→0 against adversarial logits. (Trace-level
   large constants remain a container-v1.1 nicety; seeded channels suffice.)
3. CUDA: no new kernels; run the identities under the quorum scheduler;
   measure the §3 probe set + the G6 amateur-edge probes at R ∈ {8, 32, 128}
   on Qwen3-0.6B (amateur = context-truncated same-model, a-host form).
   (a-device) lands with multi-model + the container extern-channel (G6).
4. Metal mirrors via the goldens (mac-master's existing gate discipline).
