/** @module Interface pie:core/session **/
/**
 * Sends a message to the remote user client
 */
export function send(message: string): void;
/**
 * Receives an incoming message from the remote user client
 */
export function receive(): FutureString;
/**
 * Sends a file to the remote user client
 */
export function sendFile(data: Blob): void;
/**
 * Receives an incoming file from the remote user client
 */
export function receiveFile(): FutureBlob;
export type FutureString = import('./pie-core-types.js').FutureString;
export type FutureBlob = import('./pie-core-types.js').FutureBlob;
export type Blob = import('./pie-core-types.js').Blob;
