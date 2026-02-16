// Sampler configuration — maps to pie:core/inference.Sampler WIT variant.

import type { Sampler as WitSampler } from 'pie:core/inference';

/**
 * Sampler constructors for controlling token selection during generation.
 *
 * Each method returns a WIT-compatible discriminated union value
 * that can be passed to `ForwardPass.sampler()`.
 */
export const Sampler = {
  /** Greedy decoding (temperature=0, top-k=1). */
  greedy(): WitSampler {
    return { tag: 'multinomial', val: [0.0, 1] };
  },

  /** Multinomial sampling with temperature and top-k. */
  multinomial(temperature: number, topK: number): WitSampler {
    return { tag: 'multinomial', val: [temperature, topK] };
  },

  /** Top-k sampling. */
  topK(temperature: number, topK: number): WitSampler {
    return { tag: 'top-k', val: [temperature, topK] };
  },

  /** Top-p (nucleus) sampling. */
  topP(temperature: number, topP: number): WitSampler {
    return { tag: 'top-p', val: [temperature, topP] };
  },

  /** Min-p sampling. */
  minP(temperature: number, minP: number): WitSampler {
    return { tag: 'min-p', val: [temperature, minP] };
  },

  /** Combined top-k + top-p sampling. */
  topKTopP(temperature: number, topK: number, topP: number): WitSampler {
    return { tag: 'top-k-top-p', val: [temperature, topK, topP] };
  },

  /** Embedding extraction (no sampling). */
  embedding(): WitSampler {
    return { tag: 'embedding' };
  },

  /** Full distribution output. */
  dist(temperature: number, topK: number): WitSampler {
    return { tag: 'dist', val: [temperature, topK] };
  },
} as const;

/** Re-export the WIT sampler type for external use. */
export type { WitSampler as SamplerType };
