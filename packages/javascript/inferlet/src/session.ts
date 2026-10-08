// Session functions — wraps pie:inferlet/session WIT interface.
// Handles communication with the remote user client.

import * as _session from 'pie:inferlet/session@0.3.0';

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

/** Receives a text message from the remote user client; `undefined` once
 *  the client has closed the connection.
 *
 *  Synchronous: a JS guest cannot lower the world's `async func` imports, so
 *  this goes through the host's blocking twin and the guest's task blocks
 *  until a message arrives. */
export function receive(): string | undefined {
    return _session.receiveBlocking();
}

/** Sends a file (binary data) to the remote user client. */
export function sendFile(data: Uint8Array): void {
    _session.sendFile(data);
}

/** Receives a file from the remote user client (synchronous; see `receive`). */
export function receiveFile(): Uint8Array | undefined {
    return _session.receiveFileBlocking();
}
