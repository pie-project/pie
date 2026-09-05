/** @module Interface pie:inferlet/session@0.3.0 **/
/**
 * Sends a message to the remote user client
 */
export function send(message: string): void;
/**
 * Receives an incoming message from the remote user client
 * Sends a file to the remote user client
 */
export function sendFile(data: Blob): void;
/**
 * Receives an incoming file from the remote user client
 * `receive` for a guest that cannot lower an `async func` (see
 * `channel.take-blocking`): the guest's task blocks until a message
 * arrives.
 */
export function receiveBlocking(): string | undefined;
/**
 * `receive-file` for the same guests.
 */
export function receiveFileBlocking(): Blob | undefined;
export type Blob = import('./pie-inferlet-types.js').Blob;
