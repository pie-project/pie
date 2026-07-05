// Public API for the Pie inferlet JavaScript SDK.
//
// Quickstart:
//
//     import { Context, Sampler } from 'inferlet';
//
//     const ctx = new Context();
//
//     ctx.system('You are helpful.').user('What is 2 + 2?');
//     const text = await ctx
//       .generate(Sampler.argmax(), { maxTokens: 64 })
//       .collectText();
//
// Three-layer surface:
//
// * `Context` — KV cache + chat fillers + `forward()` / `generate()`.
// * `Forward` (`ctx.forward()`) — single forward-pass primitive with auto
//   page management. For prefill / scoring / custom loops.
// * `Generator` (`ctx.generate()`) — multi-step state machine over
//   Forward. Iterate with `for await (const step of gen)`, or use
//   `await gen.collectText() / .collectTokens() / .collectJson()`.
//
// Streaming decoders for chat / reasoning / tools live as independent
// modules — compose by hand, no implicit suppression:
//
//     import { chat, reasoning, tools } from 'inferlet';
//
//     const chatDec = new chat.Decoder();
//     for await (const step of gen) {
//       const out = await step.execute();
//       const ev = chatDec.feed(out.tokens);
//       if (ev.type === 'delta') process.stdout.write(ev.text);
//       else if (ev.type === 'done') break;
//     }
//
// Constraint specs (`jsonSchema`, `anyJson`, `regex`, `ebnf`) implement
// the `Schema` interface — duck-typed, so your own grammar source class
// plugs in by adding a `buildConstraint()` method.

// ── Core ─────────────────────────────────────────────────────────────
export * as model from './model.js';
export { Adapter } from './adapter.js';
export { Context } from './context.js';

// ── Forward primitive ────────────────────────────────────────────────
export { Forward, Output } from './forward.js';
export type { SampleHandle, ProbeHandle, Brle } from './forward.js';

// ── Generator ────────────────────────────────────────────────────────
export { Generator, GenStep } from './generation.js';
export type { GenerateOptions } from './generation.js';

// ── Sampler / Probe ──────────────────────────────────────────────────
export {
  Sampler,
  Logits,
  Distribution,
  Logprob,
  Logprobs,
  Entropy,
} from './sample.js';
export type { Probe, ProbeKind } from './sample.js';

// ── Decoders + tools (sub-modules) ───────────────────────────────────
export * as chat from './chat.js';
export * as reasoning from './reasoning.js';
export * as tools from './tools.js';

// ── Constraints ──────────────────────────────────────────────────────
export {
  Grammar,
  Matcher,
  GrammarConstraint,
  jsonSchema,
  anyJson,
  regex,
  ebnf,
  grammar,
} from './grammar.js';
export type { Schema, Constraint } from './grammar.js';

// ── Speculation ──────────────────────────────────────────────────────
export type { Speculator } from './spec.js';

// ── Runtime / IO sub-modules ─────────────────────────────────────────
export * as runtime from './runtime.js';
export * as session from './session.js';
export * as messaging from './messaging.js';
export * as zo from './zo.js';

// Convenience re-exports for the most commonly subscribed-to types.
export { Subscription } from './messaging.js';
