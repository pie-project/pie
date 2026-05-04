/** @module Interface pie:core/messaging **/
/**
 * Pushes a message onto a topic queue
 */
export function push(topic: string, message: string): void;
/**
 * Pulls the next message from a topic queue
 */
export function pull(topic: string): FutureString;
/**
 * Publishes a message to a topic (broadcast to all subscribers)
 */
export function broadcast(topic: string, message: string): void;
/**
 * Subscribes to a topic and returns a subscription handle
 */
export function subscribe(topic: string): Subscription;
export type Pollable = import('./wasi-io-poll.js').Pollable;
export type FutureString = import('./pie-core-types.js').FutureString;

export class Subscription {
  /**
   * This type does not have a public constructor.
   */
  private constructor();
  /**
  * Pollable to check for new messages on the topic
  */
  pollable(): Pollable;
  /**
  * Retrieves a new message from the topic, if available
  */
  get(): string | undefined;
  /**
  * Cancels the subscription
  */
  unsubscribe(): void;
}
