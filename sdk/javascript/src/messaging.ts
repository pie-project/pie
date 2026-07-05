// Messaging functions — wraps pie:core/messaging WIT interface.

import * as _msg from 'pie:core/messaging';

/** Pushes a message onto a topic queue. */
export function push(topic: string, message: string): void {
  _msg.push(topic, message);
}

/** Pulls the next message from a topic queue. */
export function pull(topic: string): Promise<string> {
  return _msg.pull(topic);
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
 * Async iterable — use `for await...of` to consume messages. Implements
 * `Disposable` for use with `using`. Cancelling the subscription (via
 * `unsubscribe()` or scope exit) drops the underlying stream reader,
 * which unsubscribes from the topic host-side.
 */
export class Subscription implements AsyncIterable<string>, Disposable {
  readonly #reader: ReadableStreamDefaultReader<string>;

  /** @internal */
  constructor(stream: ReadableStream<string>) {
    this.#reader = stream.getReader();
  }

  /** Waits until a message arrives, then returns it. Resolves to
   *  `undefined` once the stream closes. */
  async next(): Promise<string | undefined> {
    const { value, done } = await this.#reader.read();
    return done ? undefined : value;
  }

  /** Cancels the subscription, dropping the stream reader (unsubscribe). */
  unsubscribe(): void {
    void this.#reader.cancel();
  }

  /** Disposable protocol — calls `unsubscribe()`. */
  [Symbol.dispose](): void {
    this.unsubscribe();
  }

  /** Async iterate over incoming messages (until the stream closes). */
  async *[Symbol.asyncIterator](): AsyncIterableIterator<string> {
    while (true) {
      const msg = await this.next();
      if (msg === undefined) break;
      yield msg;
    }
  }
}
