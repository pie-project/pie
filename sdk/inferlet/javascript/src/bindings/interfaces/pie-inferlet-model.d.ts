/** @module Interface pie:inferlet/model@0.3.0 **/
/**
 * The engine serves exactly one model; these are global functions over
 * that single bound model (no `model` resource handle). Identity + memory-
 * shaping capabilities only — the tokenizer surface (encode/decode/vocabs/
 * special-tokens/split-regex) lives in the sibling `tokenizer` interface.
 * Returns the name of the bound model
 */
export function name(): string;
/**
 * Returns the model architecture identifier (e.g. "gemma4", "qwen3_6")
 */
export function architecture(): string;
/**
 * Whether the bound model enables system speculation by default
 */
export function defaultSystemSpeculation(): boolean;
/**
 * The draft head's chain depth: how many tokens past each readout row
 * the `mtp_drafts` intrinsic carries (`[n_out × depth]`, row-major,
 * each row's chain conditioned on the trunk's argmax at that row). Zero
 * for a model with no draft head, in which case `mtp_drafts` and
 * `mtp_logits` are unavailable. Static, like `frame-size`.
 */
export function mtpDepth(): number;
export function draftBlock(): BlockDrafter | undefined;
export function passKind(): ForwardKind;
export function canvas(): CanvasShape | undefined;
/**
 * Logits/output dimension (= hf_config.vocab_size). May EXCEED the
 * tokenizer's vocabs() token count due to padding. Use THIS for
 * sampler-program lowering and any logits-shaped op; use vocabs() only for
 * token-space work. Sourced from the model (not hardcoded) so the
 * inferlet's lowering vocab == the engine's logits / recognizer-table vocab.
 */
export function outputVocabSize(): number;
/**
 * Tokens per KV page for the bound model/engine.
 */
export function kvPageSize(): number;
/**
 * Waves per frame (k) for this deployment — a static constant fixed at
 * engine start, exactly like kv-page-size. `forward.submit` takes exactly
 * this many ordered slots; slot i executes in wave i. Guests must be
 * output-correct for any k: it decides submission granularity and
 * resource sizing, never token-level behavior. Never renegotiated per
 * frame and never adapted from runtime timing.
 */
export function frameSize(): number;
/**
 * How long (microseconds) a pipeline may hold a frame's wait-set without
 * submitting before the engine stops waiting for it.
 * 
 * A frame does not dispatch until every participating pipeline has
 * submitted its slot, so membership is a promise to keep submitting, and
 * this is that promise's bound. The clock is per pipeline, measures
 * CONSECUTIVE silence while actually blocking a seal, and is suspended
 * whenever the engine is what the pipeline is waiting on — so it is not a
 * latency budget and a pipeline that keeps submitting can never trip it
 * however long it runs.
 * 
 * A pipeline that intends to stop — blocked on a user turn, waiting on a
 * peer, or simply done for now — calls `forward.park` instead, which
 * leaves the wait-set and stops the clock. What this bound catches is
 * therefore only the case the engine cannot otherwise distinguish: a
 * member that will never submit again.
 * 
 * Small (50ms by default) because it measures a much narrower interval
 * than its size suggests: run-ahead, an unsettled dispatch, a bind in
 * flight and `forward.park` all stop the clock, so what is left is a
 * pipeline hard-blocking the seal with nothing owed to it. It is not
 * smaller because the one-off cost of BUILDING the first pass of a loop
 * — tracing and compiling it after the prefill result lands — is on the
 * clock and was measured between 5ms and 20ms; a steady decode turnaround
 * is far below that.
 * 
 * The corollary is the one thing a pipeline has to be deliberate about:
 * work done between receiving a result and submitting the next fire is
 * ON the clock, because during it the fleet is stalled on this pipeline
 * and nothing is owed back. Anything that can outlast this bound —
 * tokenizing a large document, compiling a grammar, an RPC, a decision
 * that waits on another pipeline — should be preceded by `forward.park`
 * and is rejoined by simply submitting again. Parking is cheap and
 * unparking is implicit, so when in doubt, park.
 * 
 * Overrunning it is not fatal. The engine simply stops waiting: the slot
 * leaves the frame, already-submitted work still runs, and the next
 * submit rejoins. What it costs is a boundary — so treat this as the
 * budget that keeps you IN each frame, not as a deadline you die to.
 * (Termination is reserved for a pipeline that goes silent for orders of
 * magnitude longer without ever parking, i.e. an abandoned one.)
 * 
 * Static, like frame-size and unlike channel-capacity: guests read it
 * once and size their work against it, so it must not move.
 */
export function submitDeadlineUs(): bigint;
/**
 * Host-reader channel capacity, in cells, that lets one lane sustain the
 * engine's run-ahead without the ring becoming the bottleneck. Size every
 * host-reader channel to at least this; the staging margin is already
 * included, so guests need no arithmetic.
 * 
 * Under-sizing does not fail loudly. At k = 1 there is no frame to
 * validate, so an undersized ring simply serialises the lane with no
 * diagnostic — and because the scheduler waits for ALL lanes, a serialised
 * lane holds up every co-resident tenant, not just itself.
 * 
 * UNLIKE frame-size this is not promised to be static: it derives from the
 * host resubmit turnaround, which the runtime may later adapt. Read it,
 * don't cache it across a run.
 */
export function channelCapacity(): number;
/**
 * Fires one lane may have submitted and not yet taken -- the run-ahead
 * window, in fires. A host-reader ring of `channel-capacity` cells holds
 * exactly this many plus the visibility margin, so a guest that keeps
 * this many fires in flight never serialises and never overruns. The
 * runtime derives both numbers from one source (`engine::runahead`);
 * guests read this rather than recovering it from `channel-capacity`.
 * Not static, for the same reason `channel-capacity` is not.
 */
export function runAheadWindow(): number;
/**
 * Max embed tokens in a single pass (C) — the prefill chunk budget.
 * Guests split a prompt of L tokens into ceil(L / C) chunk passes;
 * chunking is guest-side, against this static constant.
 */
export function maxEmbedLength(): number;
/**
 * ── Working-set / arena capabilities (global, over the bound model) ──
 * Memory-shaping parameters of the bound model's engine, so an inferlet
 * can size working sets and validate fold lengths before allocating.
 * Size in bytes of one folded recurrent-state object. 0 if the model has
 * no recurrent state (pure attention).
 */
export function rsStateSize(): bigint;
/**
 * Tokens per buffered RS page. 0 if the model has no recurrent state.
 */
export function rsBufferPageSize(): number;
/**
 * Fold granularity in tokens: an RS fold of `n` tokens requires `n` to
 * be a positive multiple of this value. 1 (or 0) means unconstrained;
 * 0 also implies the model has no recurrent state. (Token-causal RS models
 * — Qwen3.5 GDN, Nemotron-H Mamba2 — report 1.)
 */
export function rsFoldGranularity(): number;
/**
 * Size in bytes of one unified-arena accounting block. In v1 the KV page is
 * exactly one block, so this is the byte size of one KV page; an RS slab
 * occupies an integer number of these blocks.
 */
export function arenaBlockSize(): bigint;
/**
 * What a guest needs to seed a BLOCK drafter's draft pass: the rows one
 * pass carries (the anchor and `rows - 1` mask slots), the id every row
 * but the first carries in, and whether the block sees itself — in which
 * case the guest states the mask that says so. Facts the head was trained
 * at, not policy: which rows to verify, and whether to draft at all, stay
 * the guest's. `none` for a model with no block drafter (a chained head
 * is `mtp-depth`). Static, like `mtp-depth`.
 */
export interface BlockDrafter {
  rows: number,
  maskToken: number,
  bidirectional: boolean,
  /**
   * The first row whose readout is a proposal: 1 when the anchor row
   * proposes nothing, 0 when every row does (the anchor's row then
   * predicts the token after it).
   */
  proposalsFrom: number,
}
/**
 * Which forward-pass kind the bound model requires. A guest may only
 * construct the `forward-pass` of the matching interface; the other
 * interfaces' constructors error immediately.
 * 
 * Do NOT derive this by parsing `architecture()` — that is an open set and
 * every new model family would break guests. This enum is closed over
 * STATE SEMANTICS, so it does not grow with the model zoo.
 * 
 * This is also the ONE place the linear/recurrent class is asked about.
 * `!= attention` means the model carries a fused recurrent state that
 * folds tokens IRREVERSIBLY (linear-attention / SSM: Qwen3.5 GDN,
 * Nemotron-H Mamba2), as opposed to pure attention whose KV is per-token
 * and reversibly discardable. That in turn pins the speculative-commit
 * CONTRACT — FOLD-COMMIT (fold only the accepted prefix, holding the
 * uncertain tail in the RS buffer; folding is runtime-managed in-forward)
 * vs KV-SLOT DISCARD — so a spec-decode loop can stay model-agnostic.
 * Do not re-derive the class from `rs-state-size() > 0`: that is a
 * byte-accounting accident that happens to agree.
 * # Variants
 * 
 * ## `"attention"`
 * 
 * Per-token, reversibly discardable KV only  -> `forward`
 * ## `"recurrent"`
 * 
 * Irreversibly folded recurrent state only   -> `forward-recurrent`
 * ## `"hybrid"`
 * 
 * Both attention and recurrent layers        -> `forward-hybrid`
 * ## `"diffusion"`
 * 
 * Paged KV plus a canvas the model denoises in place, one trunk
 * read causally or bidirectionally per pass -> `forward-diffusion`
 */
export type ForwardKind = 'attention' | 'recurrent' | 'hybrid' | 'diffusion';
/**
 * The canvas a diffusion model denoises: how many tokens one block is,
 * and the trunk's hidden width (the row width of a self-conditioning
 * signal). `none` for every other kind — the same fact as
 * `pass-kind() == diffusion`, stated once with its numbers.
 */
export interface CanvasShape {
  length: number,
  hidden: number,
  /**
   * How many `(id, weight)` taps per canvas row
   * `forward-diffusion.self-conditioning` takes.
   */
  selfCondTaps: number,
}
