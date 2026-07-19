# Pie capability map

What Pie exposes to guest programs (*inferlets*), and which class of
inference-time algorithm each primitive unlocks. This is the reference the rest
of the reports in this directory cite when they say an algorithm is
"Pie-native" or "needs an engine patch elsewhere".

Everything below is grounded in this repository, not in marketing copy — the
"Where" column points at the file that defines the surface.

---

## 1. PTIR — guest-authored, device-resident tensor programs

An inferlet *traces* a tensor program and hands the canonical PTIR bytes to
`forward-pass.program`. The program executes **on the GPU, inside the forward
pass**. This is the single biggest departure from vLLM/SGLang, where the only
guest-visible hook is a sampling-parameter struct.

### Intrinsics

| Intrinsic | Yields | Unlocks |
|---|---|---|
| `logits()` | `[n_out, vocab]` F32 LM-head logits | any custom sampler, logit arithmetic, contrastive decoding |
| `mtp_logits(k)` | `k` multi-token-prediction head rows (model-gated) | native MTP drafting, fused draft/verify |
| `hidden()` | residual stream at read-out (epilogue) | activation steering, representation-level intervention, hidden-state samplers |
| `query()` | the layer's projected query (attention tap) | query-aware KV selection, attention-score-driven eviction |
| `value_head()` | model-gated scalar value head (epilogue) | on-device value/verifier scoring for search |
| `layer()` | layer index, U32 scalar (attention tap) | per-layer logic, layer-contrastive decoding (DoLa-style) |
| `vocab()`, `page_size()` | trace-known constants | shape-correct programs |

Where: `sdk/rust/ptir-dsl/src/intrinsics.rs`

### Operator set

Elementwise/reduction/comparison/logical: `neg exp log cast add sub mul div rem
max_elem min_elem eq ne lt le gt ge and or not` (and more).
Where: `sdk/rust/ptir-dsl/src/value.rs`

> **Why it matters.** On a black-box server, "a new sampler" means a C++/CUDA
> patch and a redeploy. On Pie it is a traced guest program shipped as Wasm.
> Anything expressible as tensor math over `logits`/`hidden`/`value_head` is a
> user-space change.

---

## 2. Channels — GPU-resident queues that break the host round-trip

A `channel` is a GPU-resident bounded queue with full/empty bits. The *same*
handle is bound into a forward pass (by dense declaration index) and used by
the host for `put` / `set` / `take` / `read`. Channels can also be
**device-advanced**: the PTIR program itself contains the advance rule.

- `put` — hand a value to the device (seed, or next staged cell)
- `set` — atomically replace the committed front cell (fused take+put; used for
  *control words*, e.g. a temperature that the host retunes every step without
  re-submitting)
- `take` / `read` — move/copy a committed value back to the host, blocking on
  in-flight fires; a failed fire **poisons** the channel so the error surfaces
  at the read instead of hanging

Where: `interface/inferlet/forward.wit`

> **Why it matters.** A decode loop whose per-token state lives in a
> device-advanced channel does not pay a host round-trip per token. Stateful
> samplers (Mirostat, DRY, watermarking, entropy-adaptive temperature) and
> draft/verify loops become device-resident.

---

## 3. Explicit forward-pass construction

`forward-pass` is a builder; every geometry input is an individual channel.

| Binding | Meaning |
|---|---|
| `embed(...)` | embedding token ids + CSR row indptr |
| `attention(...)` | the **only** attention binding surface. `some(mask)` binds a channel to the PTIR `AttnMask` port; `none` omits it |
| `readout(indices)` | which positions to read out |
| `set-rs-working-sets(...)` | recurrent-state working sets, in resolved request order (hybrid / linear-attention / GDN models) |
| `program(...)` | canonical PTIR bytes + channel handles in dense declaration order |

Where: `interface/inferlet/forward.wit`

> **Why it matters.** *An arbitrary attention mask is a first-class guest
> input.* Sliding windows, attention sinks, tree/graph attention for
> multi-candidate verification, prefix-tree sharing, and logical-ancestry beam
> masks are all guest-expressible rather than engine features.

---

## 4. Frames — heterogeneous batched submission

`submit_frame(on, slots)` submits exactly `model.frame_size()` ordered slots.
Slot *i* executes in wave *i*; slots may repeat a handle (plain decode) or be
heterogeneous (prefill chunks in early slots, decode in the rest). Submission
validation is deterministic and structural — staged/device-advanced/latest-value
channel checks, and host-reader capacity checks that prevent overflow rather
than back-pressuring. Each non-no-op slot prepares and enqueues **run-ahead**.

Where: `sdk/rust/inferlet/src/ptir.rs` (`submit_frame`, `frame_size`), `interface/inferlet/forward.wit`

---

## 5. KV cache as a programmable data structure

KV is organized in fixed-size **pages**, with a *committed* vs *working*
distinction and **content-addressed sharing** of committed pages.

| Operation | Cost / semantics |
|---|---|
| `ctx.fork()` | copy-on-write clone. Committed pages shared; only working pages + divergent tokens cost memory. **O(1)** — memory and compute scale with divergent tokens, not prompt length |
| commit | promotes working pages; enables content-addressed sharing |
| `truncate` | drop tail tokens |
| snapshot `save` / `open` / `take` / `delete` | persist a token log + unflushed tail; `open` is an implicit fork; restore is a replay-prefill |
| `WorkingSet::fork` / `slice` / `discard`, `PageGrant` | manual page operations (Rust) |
| `RsWorkingSet::fork` | same for recurrent state |

Where: `website/docs/guide/context/{pages,sharing}.mdx`, `sdk/rust/inferlet/src/ptir.rs`, `sdk/rust/inferlet/src/snapshot.rs`

> **Why it matters.** Every branching meta-generation algorithm — best-of-N,
> self-consistency, Tree of Thoughts, MCTS, beam search, self-refine,
> tool-call rollback — is fundamentally "fork a prefix, explore, prune,
> maybe backtrack". On a black-box server that is either re-prefill or a
> prefix-cache you cannot address explicitly. Here it is a first-class,
> O(1), user-invoked operation.

---

## 6. Constrained decoding, programmable

| Surface | Detail |
|---|---|
| Constraint sources | `JsonSchema`, `AnyJson`, `Regex`, `Ebnf`, or a raw `Matcher`/`Grammar` |
| Mask representation | packed `u32` bitmask, one bit per token |
| Composition | `and_into` — word-wise AND, i.e. **intersection of several independent constraints** |
| Application | applied *inside* the pass (driver op `0x65 MaskApply`), a.k.a. late masking; composes with speculation |
| Termination | `is_terminated()` |

Where: `sdk/rust/inferlet/src/{constraint,mask}.rs`, `interface/inferlet/grammar.wit`, `runtime/grammar/` (Rust rewrite derived from XGrammar)

---

## 7. Speculation, programmable

- system speculation (`default_system_speculation`)
- **custom speculators** written by the inferlet
- native MTP draft → verify → accept via `mtp_logits(k)`
- self-speculation
- prompt-lookup / n-gram "cacheback" drafting
- speculation **composed with grammar constraints** in one pass
- working-vs-committed pages give **draft rollback** for free

Evidence in-repo: `runtime/engine/tests/inferlets/{selfspec,specverify,mtpverify,mtp-specdecode,mtp-native-verify,mtp-grammar}`, `tests/inferlets/{cacheback-speculative-decoding,mtp-speculative-decoding}`

---

## 8. Scheduling as a guest-visible market

Requests bid for capacity with credits; an inferlet can read the market and
override its bid. An algorithm can therefore express **its own
compute-allocation policy** — the mechanism a compute-optimal test-time-scaling
policy needs.

Where: `website/docs/guide/context/scheduling.mdx`

---

## 9. Agentic I/O inside the serving system

HTTP, filesystem (`/scratch` preopen), sessions, tool calling / MCP,
inter-inferlet **launch** and **messaging**, media (image/audio/video), speech.

Where: `interface/inferlet/{http,session,tools,media,speech,run,system}.wit`, `sdk/rust/inferlet/src/{http,tools,audio}.rs`

> **Why it matters.** Multi-agent debate, tool-augmented search, and
> long-horizon memory loops execute *next to the KV cache* instead of across a
> network boundary, so a tool call does not evict or re-prefill the context.

---

## 10. The expressiveness boundary, stated plainly

An inference-time algorithm needs an engine patch on vLLM/SGLang — but is
user-space on Pie — when it requires any of:

1. **More than logits per step** — hidden states, per-layer activations,
   attention queries, value heads.
2. **A custom attention mask** — tree verification, sinks, windows, ancestry.
3. **Explicit KV branching/backtracking** — fork, snapshot, truncate, rollback.
4. **Per-token stateful logic without a host round-trip** — device-advanced
   channels.
5. **Combining several distributions** — multiple contexts or models per step.
6. **A custom draft/verify rule** — including fusing accept/reject on device.
7. **Its own compute-allocation policy** — bidding, adaptive branch budgets.
8. **Tool/agent I/O interleaved with generation** without losing cache locality.

These eight axes are the columns of the uniqueness matrix in
`08-pie-uniqueness-matrix.md`.
