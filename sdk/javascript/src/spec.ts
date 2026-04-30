// Speculator interface for speculative-decoding drafters.
//
// Plug a `Speculator` into a `Generator` via the `speculator` option to
// drive draft tokens off your own logic.
//
// For host-driven speculation (where the runtime returns next-iter draft
// tokens via the forward-pass output's spec channel), pass
// `systemSpeculation: true` to `ctx.generate(...)` instead — that mode
// is built into the Generator and does not need a Speculator
// implementation.

/** A speculative-decoding drafter.
 *
 *  Each iteration the `Generator` asks for `draft()` tokens, runs the
 *  verifier, then reports `accept()`. On rejection the Generator calls
 *  `rollback()` so the speculator can truncate any state it grew during
 *  drafting.
 *
 *  User-defined speculators don't need to inherit — any object with the
 *  matching methods satisfies this interface (TS structural typing):
 *
 *      class MyDrafter implements Speculator {
 *        draft() { return [tokens, positions] as const; }
 *        accept(tokens) { ... }
 *        rollback(n) { ... }
 *        reset() { ... }
 *      }
 */
export interface Speculator {
  /** Produce draft tokens and their absolute positions for the next
   *  forward pass. Return `[new Uint32Array(), new Uint32Array()]` for
   *  "no speculation this step". */
  draft(): readonly [Uint32Array, Uint32Array];

  /** Called with the verifier's accepted token sequence. The first
   *  accepted token corresponds to the anchor's own next-token
   *  prediction; the rest (if any) are matched drafts. */
  accept(tokens: Uint32Array): void;

  /** Roll back the last `n` drafted tokens — used when the verifier
   *  rejects the tail of the draft sequence and the speculator's own
   *  internal context needs to mirror that truncation. */
  rollback(n: number): void;

  /** Reset to initial state. */
  reset(): void;
}
