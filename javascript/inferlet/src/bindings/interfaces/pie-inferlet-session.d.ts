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
 * `send-file` with a suggested file name, so what arrives at the client
 * is a named file rather than an anonymous blob.
 * 
 * The bytes are already a guest value here -- that is what makes this
 * `send-file` and not `send-frames`. What it adds is the one thing
 * `send-file` cannot say: WHAT THE FILE IS. `pie run -o DIR` writes an
 * unnamed file as `file-0000.bin`, which is the right answer for a blob
 * nobody described and the wrong one for a guest that knows it just
 * produced `image.latent.f32`.
 * 
 * `name` is a suggestion and not a path, exactly as in `send-frames`:
 * the host keeps the last component and the client sanitises again.
 */
export function sendFileAs(data: Blob, name: string): void;
/**
 * Receives an incoming file from the remote user client
 * Encodes `f` and streams the bytes to the client under `name`, WITHOUT
 * the encoded file ever entering WASM linear memory.
 * 
 * This is `send-file` for a handle rather than a buffer, and the
 * difference is the whole point: `send-file(f.encode(format))` produces
 * the same file having first copied every byte of it into the guest's
 * heap. A 121-frame clip is tens of megabytes; a guest that only wants
 * the client to have it should never hold it.
 * 
 * `name` is a suggested file name, carried beside the bytes to the
 * client (`pie run -o DIR` writes it there under this name). It is a
 * suggestion, not a path: the client sanitises it.
 */
export function sendFrames(f: Frames, format: ImageFormat, name: string): void;
/**
 * `send-frames` for audio. Same statute: the samples are encoded
 * host-side and streamed out, never materialized in linear memory.
 */
export function sendPcm(p: Pcm, format: AudioFormat, name: string): void;
/**
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
export type Error = import('./pie-inferlet-types.js').Error;
export type Frames = import('./pie-inferlet-frames.js').Frames;
export type Pcm = import('./pie-inferlet-frames.js').Pcm;
export type ImageFormat = import('./pie-inferlet-frames.js').ImageFormat;
export type AudioFormat = import('./pie-inferlet-frames.js').AudioFormat;
