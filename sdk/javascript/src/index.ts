// Public API for the Pie inferlet JavaScript SDK.
//
// STATUS: the forward-pass surface is NOT available from JavaScript yet.
//
// The runtime's guest-facing forward-pass surface was replaced. The old
// `pie:core/inference` interface, which exposed a fixed host-side sampler
// (`Sampler.argmax()`, probes, `Generator`), is gone. In its place the guest
// traces a program and ships canonical PTIR container bytes through one of
// `pie:inferlet/forward`, `forward-recurrent`, or `forward-hybrid`.
//
// Nothing in this package can produce those bytes. The Rust SDK does it with
// `compiler/dsl` (the tracing eDSL) and `compiler/ir` (the container encoder);
// neither has a JavaScript counterpart, and the encoder has to agree with the
// Rust one byte for byte. Porting it is its own project — see
// `forward_refactor.md` section 10.4 and the tracking note in
// `scripts/check-sdk-interfaces.sh`.
//
// So the modules that were built on the removed interface — `context`,
// `forward`, `generation`, `sample`, `grammar`, `tools`, `adapter`,
// `runtime`, `zo`, `spec` — have been deleted rather than left importing a
// surface that no longer exists. They were not usable; they only looked
// usable.
//
// What is here is the part of the SDK that survives the move unchanged: the
// non-forward interfaces, whose WIT definitions came through the split
// intact.
//
//     import { model, session, chat, reasoning } from 'inferlet';
//
// `grammar` and `tools` DO exist as interfaces in the new world, but under
// different shapes (`pie:inferlet/grammar`, `pie:inferlet/tools`) than the
// deleted modules of those names targeted. They come back with the port, not
// before.
//
// The generated `bindings/` tree is current: `npm run generate-bindings`
// regenerates it from `interface/inferlet/`, and every interface in the world
// above is present there. Only the hand-written layer is missing.

export * as model from './model.js';
export * as tokenizer from './tokenizer.js';
export * as session from './session.js';
export * as chat from './chat.js';
export * as reasoning from './reasoning.js';
