// ForwardPass wrapper over pie:core/inference WIT resource.

import {
  ForwardPass as _ForwardPass,
} from 'pie:core/inference';
import type {
  Sampler,
  Brle,
  Output,
} from 'pie:core/inference';
import type { Context } from 'pie:core/context';
import type { Model } from 'pie:core/model';
import type { Adapter } from 'pie:core/adapter';
import { awaitFuture } from './_async.js';

/**
 * A single forward pass through the model.
 *
 * Wraps the `pie:core/inference.ForwardPass` WIT resource.
 * Configure inputs, outputs, and sampling, then call `execute()`.
 */
export class ForwardPass {
  /** @internal */
  readonly _handle: _ForwardPass;

  constructor(model: Model) {
    this._handle = new _ForwardPass(model);
  }

  /** Bind a context (KV cache) to this forward pass. */
  context(ctx: Context): void {
    this._handle.context(ctx);
  }

  /** Set input token IDs and their position IDs. */
  inputTokens(tokens: Uint32Array, positions: Uint32Array): void {
    this._handle.inputTokens(tokens, positions);
  }

  /** Set speculative input tokens and positions. */
  inputSpeculativeTokens(tokens: Uint32Array, positions: Uint32Array): void {
    this._handle.inputSpeculativeTokens(tokens, positions);
  }

  /** Enable/disable speculative token output (enabled by default). */
  outputSpeculativeTokens(flag: boolean): void {
    this._handle.outputSpeculativeTokens(flag);
  }

  /** Set a custom attention mask (BRLE-encoded). Falls back to causal mask if not set. */
  attentionMask(mask: Brle[]): void {
    this._handle.attentionMask(mask);
  }

  /** Set a logit mask (BRLE-encoded). Falls back to all-ones if not set. */
  logitMask(mask: Brle): void {
    this._handle.logitMask(mask);
  }

  /** Configure sampling for token selection at given indices. */
  sampler(indices: Uint32Array, sampler: Sampler): void {
    this._handle.sampler(indices, sampler);
  }

  /** Attach a LoRA adapter to this forward pass. */
  adapter(adapter: Adapter): void {
    this._handle.adapter(adapter);
  }

  /** Execute the forward pass. */
  async execute(): Promise<Output> {
    const future = this._handle.execute();
    return awaitFuture(future, 'ForwardPass.execute() returned undefined');
  }
}

/** Re-export WIT output types for convenience. */
export type { Output, Brle };
