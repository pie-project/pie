// Messaging functions — wraps pie:core/messaging WIT interface.

import * as _msg from 'pie:core/messaging';
import type { Subscription as _Subscription } from 'pie:core/messaging';
import { awaitFuture } from './_async.js';

/** Pushes a message onto a topic queue. */
export function push(topic: string, message: string): void {
  _msg.push(topic, message);
}

/** Pulls the next message from a topic queue. */
export async function pull(topic: string): Promise<string> {
  return awaitFuture(_msg.pull(topic), 'pull() returned undefined');
}

/** Broadcasts a message to all subscribers of a topic. */
export function broadcast(topic: string, message: string): void {
  _msg.broadcast(topic, message);
}

/** Subscribes to a topic, returning a `Subscription` handle. */
export function subscribe(topic: string): Subscription {
  return new Subscription(_msg.subscribe(topic));
}

/**
 * A subscription to a broadcast topic.
 *
 * Async iterable — use `for await...of` to consume messages.
 * Implements `Disposable` for use with `using`.
 */
export class Subscription implements AsyncIterable<string>, Disposable {
  /** @internal */
  readonly _handle: _Subscription;

  /** @internal */
  constructor(handle: _Subscription) {
    this._handle = handle;
  }

  /** Waits until a message arrives, then returns it. */
  async next(): Promise<string | undefined> {
    const pollable = this._handle.pollable();
    while (!pollable.ready()) {
      pollable.block();
    }
    return this._handle.get();
  }

  /** Cancels the subscription. */
  unsubscribe(): void {
    this._handle.unsubscribe();
  }

  /** Disposable protocol — calls `unsubscribe()`. */
  [Symbol.dispose](): void {
    this.unsubscribe();
  }

  /** Async iterate over incoming messages (infinite — break manually). */
  async *[Symbol.asyncIterator](): AsyncIterableIterator<string> {
    while (true) {
      const msg = await this.next();
      if (msg === undefined) break;
      yield msg;
    }
  }
}
