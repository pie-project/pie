// Samplers and probes for forward-pass output slots.
//
// The host's `forward-pass.sampler` slot folds two unrelated concerns into
// a single WIT variant. This module keeps them distinct:
//
// * `Sampler` picks a token. Use as `forward.sample(indices, Sampler.argmax())`.
// * Probes (`Logits`, `Distribution`, `Logprob`, `Logprobs`, `Entropy`) read
//   shape information without sampling. Use as
//   `forward.probe(index, Distribution(1.0, 32))`.
//
// Both attach to the same forward-pass slot, so a single `Forward` can mix
// samplers and probes freely.

import type { Sampler as WitSampler } from 'pie:core/inference';

// =============================================================================
// Sampler — token-producing
// =============================================================================

/**
 * Token-producing sampler — opaque wrapper over the WIT sampler variant.
 *
 * Construct via the `Sampler` namespace constructors — never directly.
 * The internal WIT variant is exposed as `_variant` for the SDK; user
 * code should treat `Sampler` as opaque.
 */
export interface Sampler {
  readonly _variant: WitSampler;
}

export const Sampler = {
  /** Greedy / argmax — deterministic. Recommended for grammar-constrained
   *  generation where most masked positions have only a handful of valid
   *  tokens and stochastic sampling rarely helps. */
  argmax(): Sampler {
    return { _variant: { tag: 'top-p', val: [0.0, 1.0] } };
  },

  /** Top-p (nucleus) sampling. `temperature = 0.0` collapses to argmax. */
  topP(temperature: number, p: number): Sampler {
    return { _variant: { tag: 'top-p', val: [temperature, p] } };
  },

  /** Top-k sampling: sample from the top `k` tokens by probability. */
  topK(temperature: number, k: number): Sampler {
    return { _variant: { tag: 'top-k', val: [temperature, k] } };
  },

  /** Min-p sampling: keep tokens with `probability >= p * max_prob`. */
  minP(temperature: number, p: number): Sampler {
    return { _variant: { tag: 'min-p', val: [temperature, p] } };
  },

  /** Combined top-k + top-p: first restrict to top `k`, then nucleus `p`. */
  topKTopP(temperature: number, k: number, p: number): Sampler {
    return { _variant: { tag: 'top-k-top-p', val: [temperature, k, p] } };
  },

  /** Plain multinomial after temperature scaling. `draws` is a per-sample
   *  multiplier (typically 1). */
  multinomial(temperature: number, draws: number = 1): Sampler {
    return { _variant: { tag: 'multinomial', val: [temperature, draws] } };
  },
} as const;


// =============================================================================
// Probes — distribution access (no sampling)
// =============================================================================
//
// Each probe is a tagged interface that doubles as both spec and runtime
// marker. `forward.probe(idx, X)` consumes one of these and returns a handle
// whose `kind` selects which `output.*` accessor decodes the result.

/** Pre-softmax, untemperatured logits as packed native-endian f32 bytes
 *  (length `vocab_size * 4`). Decode via
 *  `new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)`. */
export interface Logits {
  readonly kind: 'logits';
}

/** Top-`k` token IDs paired with probabilities (post-softmax,
 *  temperature-scaled). `k = 0` returns the full vocabulary. */
export interface Distribution {
  readonly kind: 'distribution';
  readonly temperature: number;
  readonly k: number;
}

/** `log p(token | context)` at this position, no temperature scaling. */
export interface Logprob {
  readonly kind: 'logprob';
  readonly token: number;
}

/** `log p(t | context)` for each `t` in `tokens`, no temperature scaling. */
export interface Logprobs {
  readonly kind: 'logprobs';
  readonly tokens: Uint32Array;
}

/** Shannon entropy `H(p) = -sum(p log p)` of the unscaled distribution. */
export interface Entropy {
  readonly kind: 'entropy';
}

/** Union of all probe kinds. */
export type Probe = Logits | Distribution | Logprob | Logprobs | Entropy;


// ─── Probe constructors ────────────────────────────────────────────────────

export function Logits(): Logits {
  return { kind: 'logits' };
}

export function Distribution(temperature: number = 1.0, k: number = 0): Distribution {
  return { kind: 'distribution', temperature, k };
}

export function Logprob(token: number): Logprob {
  return { kind: 'logprob', token };
}

export function Logprobs(tokens: Iterable<number>): Logprobs {
  return { kind: 'logprobs', tokens: new Uint32Array(tokens) };
}

export function Entropy(): Entropy {
  return { kind: 'entropy' };
}


// =============================================================================
// Internal: probe → WIT sampler conversion
// =============================================================================

/** @internal Lower a probe spec into the WIT sampler variant. */
export function _probeToWit(probe: Probe): WitSampler {
  switch (probe.kind) {
    case 'logits':       return { tag: 'raw-logits' };
    case 'distribution': return { tag: 'dist', val: [probe.temperature, probe.k] };
    case 'logprob':      return { tag: 'logprob', val: probe.token };
    case 'logprobs':     return { tag: 'logprobs', val: probe.tokens };
    case 'entropy':      return { tag: 'entropy' };
  }
}

/** @internal Map a probe to its accessor-kind string (matches `Output.*`). */
export function _probeAccessorKind(probe: Probe): ProbeKind {
  // For Logprob and Logprobs the host produces the same SlotOutput shape
  // (a logprobs slot), so they share the accessor kind.
  if (probe.kind === 'logprob') return 'logprobs';
  return probe.kind;
}

/** @internal Discriminator for typed probe handles. */
export type ProbeKind = 'logits' | 'distribution' | 'logprobs' | 'entropy';

/** @internal Pull the accessor-kind discriminator out of a `Probe` variant
 *  so `forward.probe(idx, Distribution(...))` returns
 *  `ProbeHandle<'distribution'>`. `Logprob` and `Logprobs` share the
 *  `'logprobs'` accessor since the host produces the same slot shape. */
export type ProbeKindOf<P extends Probe> = P extends Logprob | Logprobs
  ? 'logprobs'
  : P['kind'] extends ProbeKind
  ? P['kind']
  : never;
