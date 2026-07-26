/** @module Interface pie:inferlet/session@0.3.0 **/
/**
 * Sends a message to the remote user client
 */
export function send(message: string): void;
/**
 * Receives an incoming message from the remote user client
 */
export function receive(): Promise<string | undefined>;
/**
 * Sends a file to the remote user client
 */
export function sendFile(data: Blob): void;
/**
 * Receives an incoming file from the remote user client
 */
export function receiveFile(): Promise<Blob | undefined>;
export type Blob = import('./pie-inferlet-types.js').Blob;
