// Session functions — wraps pie:core/session WIT interface.
// Handles communication with the remote user client.

import * as _session from 'pie:core/session';
import { awaitFuture } from './_async.js';

/** Sends a text message to the remote user client. */
export function send(message: string): void {
    _session.send(message);
}

/** Receives a text message from the remote user client. */
export async function receive(): Promise<string> {
    return awaitFuture(_session.receive(), 'receive() returned undefined');
}

/** Sends a file (binary data) to the remote user client. */
export function sendFile(data: Uint8Array): void {
    _session.sendFile(data);
}

/** Receives a file from the remote user client. */
export async function receiveFile(): Promise<Uint8Array> {
    return awaitFuture(_session.receiveFile(), 'receiveFile() returned undefined');
}
