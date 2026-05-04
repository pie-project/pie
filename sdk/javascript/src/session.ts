// Session functions — wraps pie:core/session WIT interface.
// Handles communication with the remote user client.

import * as _session from 'pie:core/session';
import { awaitFuture } from './_async.js';

/** Sends a message to the remote user client.
 *
 * Strings are sent verbatim. Anything else is JSON-stringified
 * (objects, arrays, numbers, bools).
 *
 * ```ts
 * session.send("plain text");
 * session.send({ event: "tick", n: 3 });
 * session.send([1, 2, 3]);
 * ```
 */
export function send(message: unknown): void {
    if (typeof message === 'string') {
        _session.send(message);
    } else {
        _session.send(JSON.stringify(message));
    }
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
