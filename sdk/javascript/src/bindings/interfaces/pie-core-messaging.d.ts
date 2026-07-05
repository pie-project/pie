/** @module Interface pie:core/messaging **/
/**
 * Pushes a message onto a topic queue
 */
export function push(topic: string, message: string): void;
/**
 * Pulls the next message from a topic queue
 */
export function pull(topic: string): Promise<string>;
/**
 * Publishes a message to a topic (broadcast to all subscribers)
 */
export function broadcast(topic: string, message: string): void;
/**
 * Subscribes to a topic; the returned stream yields each broadcast
 * message. Dropping the stream reader unsubscribes.
 */
export function subscribe(topic: string): ReadableStream<string>;
