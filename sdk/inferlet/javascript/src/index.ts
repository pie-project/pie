// Public API for the Pie inferlet JavaScript SDK.
//
// An inferlet is a small program that runs next to the model. The forward-pass
// surface (`pie:inferlet/forward*`) takes a traced ETA program; `eta` traces
// it from ordinary JavaScript and encodes the canonical container bytes — the
// same bytes the Rust SDK's `eta-dsl` emits for the same program, so a JS
// inferlet and a Rust inferlet share the host's program cache.
//
//     import { model, session, chat, eta, grammar, mask } from '@pie-project/inferlet';
//
// A JS guest is built against the derived world in `wit/` (see
// `scripts/derive-js-wit.mjs`): componentize-js cannot lower the runtime's
// `async func` imports, so host readback goes through the blocking twins
// (`channel.take-blocking`, `session.receive-blocking`) and the guest's task
// blocks in the call. `bakery build` (or `pie build`) componentizes a JS
// inferlet with componentize-js against that world.

export * as model from './model.js';
export * as tokenizer from './tokenizer.js';
export * as session from './session.js';
export * as chat from './chat.js';
export * as reasoning from './reasoning.js';
export * as eta from './eta/index.js';
export * as grammar from './grammar.js';
export * as mask from './mask.js';
export * as tools from './tools.js';
export * as media from './media.js';
