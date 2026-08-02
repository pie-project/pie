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
 * Whether the bound model is linear/recurrent — i.e. it carries a fused
 * recurrent state that folds tokens IRREVERSIBLY (linear-attention / SSM:
 * Qwen3.5 GDN, Nemotron-H Mamba2), as opposed to pure attention whose KV is
 * per-token and reversibly discardable. TRUE iff the model has recurrent
 * state (equivalently `rs-state-size() > 0`). The runtime uses this to pick
 * the speculative-commit strategy — FOLD-COMMIT (fold only the accepted
 * prefix, holding the uncertain tail in the RS buffer; folding is
 * runtime-managed in-forward) for linear models vs KV-SLOT DISCARD for
 * attention — so the inferlet's spec-decode loop can stay model-agnostic.
 * Prefer this over
 * reading `rs-state-size()` as a class flag: this pins the commit CONTRACT
 * (irreversible fold ⇒ fold-commit), not a byte-accounting accident.
 */
export function isLinear(): boolean;
export function passKind(): ForwardKind;
/**
 * Logits/output dimension (= hf_config.vocab_size). May EXCEED the
 * tokenizer's vocabs() token count due to padding. Use THIS for
 * sampler-program lowering and any logits-shaped op; use vocabs() only for
 * token-space work. Sourced from the model (not hardcoded) so the
 * inferlet's lowering vocab == the driver's logits / recognizer-table vocab.
 */
export function outputVocabSize(): number;
/**
 * Tokens per KV page for the bound model/driver.
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
 * Max embed tokens in a single pass (C) — the prefill chunk budget.
 * Guests split a prompt of L tokens into ceil(L / C) chunk passes;
 * chunking is guest-side, against this static constant.
 */
export function maxEmbedLength(): number;
/**
 * ── Working-set / arena capabilities (global, over the bound model) ──
 * Memory-shaping parameters of the bound model's driver, so an inferlet
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
 * Which forward-pass kind the bound model requires. A guest may only
 * construct the `forward-pass` of the matching interface; the other
 * interfaces' constructors error immediately.
 * 
 * Do NOT derive this by parsing `architecture()` — that is an open set and
 * every new model family would break guests. This enum is closed over
 * STATE SEMANTICS, so it does not grow with the model zoo.
 * 
 * Relationship to `is-linear`: `is-linear()` == (`pass-kind()` !=
 * attention). `is-linear` remains as the commit-policy hint (irreversible
 * fold ⇒ FOLD-COMMIT); `pass-kind` is what selects the binding surface.
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
 */
export type ForwardKind = 'attention' | 'recurrent' | 'hybrid';
