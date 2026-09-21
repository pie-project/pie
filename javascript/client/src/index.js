/**
 * @file pie-client.js
 * A JavaScript client library for the Pie WebSocket server (Protocol v2).
 *
 * @requires msgpack-lite
 * @requires blake3
 */

import msgpack from 'msgpack-lite';
import { blake3 } from '@noble/hashes/blake3.js';
import { bytesToHex } from '@noble/hashes/utils.js';

/**
 * A simple asynchronous queue.
 */
class AsyncQueue {
    constructor() {
        this._values = [];
        this._resolvers = [];
    }

    put(value) {
        if (this._resolvers.length > 0) {
            const resolve = this._resolvers.shift();
            resolve(value);
        } else {
            this._values.push(value);
        }
    }

    get() {
        return new Promise((resolve) => {
            if (this._values.length > 0) {
                resolve(this._values.shift());
            } else {
                this._resolvers.push(resolve);
            }
        });
    }

    isEmpty() {
        return this._values.length === 0;
    }
}

const CHUNK_SIZE = 256 * 1024; // 256 KiB

function hashBytes(bytes) {
    return bytesToHex(blake3(bytes));
}

/**
 * One complete file an inferlet sent, with the name it suggested.
 *
 * A `Buffer` SUBCLASS, deliberately. The `file` event has always carried the
 * payload and nothing else, so every program written against this client does
 * `const [kind, data] = await proc.recv()` and then writes `data`. Returning
 * an object with `{data, name}` would break all of them to add one field.
 * Subclassing means the payload is still a Buffer -- `data.length`,
 * `fs.writeFileSync(p, data)`, `data.subarray(0, 8)` all unchanged -- and
 * `data.name` is the new fact beside it.
 *
 * `name` is `null` for `session.send-file`, which carries no name, and a
 * string for `session.send-file-as`, `send-frames` and `send-pcm`, which do.
 * It is a SUGGESTION off the wire: run it through `fileName()` before joining
 * it to a directory.
 */
const Bytes = globalThis.Buffer ?? Uint8Array;

function concatBytes(chunks) {
    if (globalThis.Buffer) return Buffer.concat(chunks);
    const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
    let at = 0;
    for (const c of chunks) { out.set(c, at); at += c.length; }
    return out;
}

export class ReceivedFile extends Bytes {
    /**
     * @param {Buffer} data The file's bytes.
     * @param {string|null} name The name the inferlet suggested, or null.
     * @returns {ReceivedFile}
     */
    // `wrap`, not `from`: `Buffer.from` is inherited here and means something
    // else, and a static that shadows it with different semantics is a trap.
    static wrap(data, name) {
        // `Object.setPrototypeOf` rather than `new`: Buffer's constructor is
        // deprecated and `Buffer.concat` is what produced these bytes, so the
        // subclass is put on afterwards rather than copying a megabyte to
        // construct it a second time.
        const file = Object.setPrototypeOf(data, ReceivedFile.prototype);
        file.name = name ?? null;
        return file;
    }

    /**
     * A name safe to join onto a directory: the directory part, `..` and NUL
     * stripped, with `fallback` standing in when nothing usable is left.
     *
     * The server sanitises on the way out too; this is the check that counts,
     * because a client does not get to assume the server it is talking to is
     * the one that wrote that code.
     *
     * @param {string} fallback
     * @returns {string}
     */
    fileName(fallback = 'file.bin') {
        const raw = this.name || '';
        const tail = raw.replace(/\\/g, '/').split('/').pop();
        const base = tail.replace(/\0/g, '').trim();
        if (base === '' || base === '.' || base === '..') return fallback;
        return base;
    }
}

/**
 * Represents a running process on the server.
 */
/**
 * A JavaScript function readied to run on pie as an inferlet: its source
 * (the function's own text, as an ES module exporting it) and the manifest
 * that names it. Built by `inferlet()`.
 */
export class Inferlet {
    constructor({ fn, name, version, entry, source, description }) {
        Object.assign(this, { fn, name, version, entry, source, description });
    }

    /** `name@version`, the id the server launches by. */
    get program() {
        return `${this.name}@${this.version}`;
    }

    manifestToml() {
        const q = (s) => JSON.stringify(String(s));
        const lines = ['[package]', `name = ${q(this.name)}`, `version = ${q(this.version)}`];
        if (this.description) lines.push(`description = ${q(this.description)}`);
        lines.push('', '[runtime]', 'language = "javascript"', `entry = ${q(this.entry)}`);
        return lines.join('\n') + '\n';
    }
}

/**
 * Mark a function as an inferlet. Its *source* travels (`fn.toString()`),
 * wrapped as a module that exports it, so it must be self-contained: it
 * reaches the inferlet library through the `pie` global the language component provides
 * (`pie.model.encode(...)`, `pie.eta`, ...) and everything else through
 * its one `input` argument; a closure over the caller's variables is a
 * ReferenceError on the server. The version is a hash of the source, so
 * an edit is a new program and an unchanged one is never re-uploaded.
 * @param {Function} fn
 * @param {{name?: string, version?: string, description?: string}} [options]
 * @returns {Inferlet}
 */
export function inferlet(fn, options = {}) {
    if (typeof fn !== 'function') throw new TypeError('inferlet() takes a function');
    const text = fn.toString();
    // A method shorthand (`{ main(input) {} }`) has no `function` keyword
    // and would not stand alone; an arrow or a function expression does.
    const isPlain = /^(async\s+)?function\b/.test(text) || /^(async\s*)?(\(|[A-Za-z_$][\w$]*\s*=>)/.test(text);
    if (!isPlain) throw new TypeError('inferlet() takes a function declaration, function expression or arrow function');
    const entry = 'main';
    const source = `export const ${entry} = ${text};\n`;
    const name = options.name ?? (fn.name || 'inferlet').replace(/_/g, '-');
    const digest = hashBytes(new TextEncoder().encode(source));
    const version = options.version ?? `0.${parseInt(digest.slice(0, 4), 16)}.${parseInt(digest.slice(4, 8), 16)}`;
    return new Inferlet({ fn, name, version, entry, source, description: options.description ?? null });
}

export class Process {
    /**
     * @param {PieClient} client The PieClient that owns this process.
     * @param {string} processId The UUID of the process.
     */
    constructor(client, processId) {
        this.client = client;
        this.processId = processId;
        this.eventQueue = client.processEventQueues.get(processId);
        if (!this.eventQueue) {
            throw new Error(`Internal error: No event queue for process ${processId}`);
        }
    }

    /**
     * Sends a signal/message to the process (fire-and-forget).
     * @param {string} message The message to send.
     */
    async signal(message) {
        await this.client.signalProcess(this.processId, message);
    }

    /**
     * Transfers a file to the process (fire-and-forget, chunked).
     * @param {Uint8Array|Buffer} fileBytes The file data to transfer.
     */
    async transferFile(fileBytes) {
        await this.client._transferFile(this.processId, fileBytes);
    }

    /**
     * Receives an event from the process. Blocks until an event is available.
     * @returns {Promise<{event: string, value: string|Uint8Array}>}
     */
    async recv() {
        if (!this.eventQueue) {
            throw new Error("Event queue is not available for this process.");
        }
        const [event, value] = await this.eventQueue.get();
        return { event, value };
    }

    /**
     * Drains the process to its end and returns what it returned: parsed
     * JSON when the return is JSON, else the string. Stdout and messages are
     * collected on `this.output`; an error rejects.
     * @returns {Promise<any>}
     */
    async result() {
        this.output ??= [];
        for (;;) {
            const { event, value } = await this.recv();
            if (event === 'stdout' || event === 'message') this.output.push(value);
            else if (event === 'return') {
                if (!value) return null;
                try { return JSON.parse(value); } catch { return value; }
            } else if (event === 'error') throw new Error(value);
        }
    }

    /**
     * Requests termination of the process.
     */
    async terminate() {
        await this.client.terminateProcess(this.processId);
    }
}

/**
 * An asynchronous client for interacting with the Pie WebSocket server.
 */
/** `serverUri` with the gateway's `/v1/ws` path when it names none. */
export function withUpgradePath(serverUri) {
    let url;
    try {
        url = new URL(serverUri);
    } catch {
        return serverUri;
    }
    if (url.pathname.replace(/\/+$/, '') === '') {
        url.pathname = '/v1/ws';
    }
    return url.toString();
}

export class PieClient {
    /**
     * @param {string} serverUri e.g. `ws://127.0.0.1:8080`; a URI with no path
     *   gets the gateway's `/v1/ws` upgrade path appended, as the Python
     *   client does.
     * @param {{WebSocket?: typeof WebSocket, identity?: string}} [options]
     *   `identity` is the value of the `x-pie-identity` trust-edge header
     *   the gateway requires (`gateway/src/ingress/identity.rs`); a
     *   deployment terminates identity at an edge proxy, a direct client
     *   must supply it. The browser's WebSocket cannot set headers, so a
     *   direct client passes a header-capable implementation (Node's `ws`)
     *   as `WebSocket` along with `identity`.
     */
    constructor(serverUri, { WebSocket: Socket = globalThis.WebSocket, identity = null } = {}) {
        this.serverUri = withUpgradePath(serverUri);
        this.Socket = Socket;
        this.identity = identity;
        this.ws = null;
        this.corrIdCounter = 0;
        this.pendingRequests = new Map();
        this.processEventQueues = new Map();
        this.pendingDownloads = new Map();
        this.orphanEvents = new Map();
        this.connectionPromise = null;
    }

    /**
     * Establishes a WebSocket connection.
     * @returns {Promise<void>}
     */
    connect() {
        if (this.ws && this.ws.readyState === this.Socket.OPEN) {
            return Promise.resolve();
        }
        if (this.connectionPromise) {
            return this.connectionPromise;
        }

        this.connectionPromise = new Promise((resolve, reject) => {
            try {
                // The WHATWG constructor takes protocols, not options; only a
                // header-capable implementation (Node's `ws`) gets the identity.
                this.ws = this.identity != null && this.Socket !== globalThis.WebSocket
                    ? new this.Socket(this.serverUri, { headers: { 'x-pie-identity': this.identity } })
                    : new this.Socket(this.serverUri);
                this.ws.binaryType = 'blob';

                this.ws.onopen = () => {
                    this._listen();
                    resolve();
                };

                this.ws.onerror = (error) => {
                    reject(new Error("WebSocket connection failed."));
                    this._rejectPendingRequests(new Error("WebSocket connection failed."));
                    this.connectionPromise = null;
                };

                this.ws.onclose = () => {
                    this._rejectPendingRequests(new Error("WebSocket connection closed."));
                    this.ws = null;
                    this.connectionPromise = null;
                };
            } catch (error) {
                reject(error);
                this.connectionPromise = null;
            }
        });

        return this.connectionPromise;
    }

    /** @private */
    async _listen() {
        this.ws.onmessage = async (event) => {
            if (event.data instanceof Blob) {
                try {
                    const arrayBuffer = await event.data.arrayBuffer();
                    const message = msgpack.decode(new Uint8Array(arrayBuffer));
                    await this._processServerMessage(message);
                } catch (e) {
                    console.error("[PieClient] Failed to decode messagepack:", e);
                }
            }
        };
    }

    /**
     * Routes incoming server messages (3 types: response, process_event, file).
     * @private
     */
    async _processServerMessage(message) {
        const msgType = message.type;

        if (msgType === 'response') {
            const { corr_id, ok, result } = message;
            if (this.pendingRequests.has(corr_id)) {
                const promiseControls = this.pendingRequests.get(corr_id);
                promiseControls.resolve({ ok, result });
                this.pendingRequests.delete(corr_id);
            }
        } else if (msgType === 'process_event') {
            const { process_id, event, value } = message;
            const eventTuple = [event, value || ''];

            if (this.processEventQueues.has(process_id)) {
                this.processEventQueues.get(process_id).put(eventTuple);
                // Clean up on terminal events
                if (event === 'return' || event === 'error') {
                    this.processEventQueues.delete(process_id);
                }
            } else {
                // Buffer orphan events
                if (!this.orphanEvents.has(process_id)) {
                    this.orphanEvents.set(process_id, []);
                }
                this.orphanEvents.get(process_id).push(eventTuple);
            }
        } else if (msgType === 'file') {
            await this._handleFileChunk(message);
        }
    }

    /** @private */
    async _handleFileChunk(message) {
        const { process_id, file_hash, chunk_index, total_chunks, chunk_data, name } = message;

        if (!this.processEventQueues.has(process_id)) return;

        if (!this.pendingDownloads.has(file_hash)) {
            if (chunk_index !== 0) return;
            this.pendingDownloads.set(file_hash, {
                buffer: [],
                totalChunks: total_chunks,
                processId: process_id,
                // The name rides on every chunk; the first is the one that is
                // guaranteed to have been seen, so it is the one remembered.
                // Undefined when the inferlet named nothing -- and undefined
                // is also what a server built before the field existed sends,
                // which is why this reads as "no name" and not as an error.
                name: name ?? null,
            });
        }

        const download = this.pendingDownloads.get(file_hash);
        download.buffer.push(chunk_data);

        if (chunk_index === total_chunks - 1) {
            this.pendingDownloads.delete(file_hash);
            const completeData = concatBytes(download.buffer);
            const computedHash = hashBytes(completeData);
            if (computedHash === file_hash && this.processEventQueues.has(download.processId)) {
                const file = ReceivedFile.wrap(completeData, download.name);
                this.processEventQueues.get(download.processId).put(['file', file]);
            }
        }
    }

    /**
     * Gracefully closes the WebSocket connection.
     */
    async close() {
        return new Promise((resolve) => {
            if (!this.ws || this.ws.readyState !== this.Socket.OPEN) {
                this._rejectPendingRequests(new Error("WebSocket connection closed."));
                resolve();
                return;
            }

            const ws = this.ws;
            const previousOnClose = ws.onclose;
            ws.onclose = (event) => {
                if (previousOnClose) previousOnClose.call(ws, event);
                resolve();
            };
            ws.close();
        });
    }

    /** @private */
    _rejectPendingRequests(error) {
        for (const { reject } of this.pendingRequests.values()) {
            reject(error);
        }
        this.pendingRequests.clear();
    }

    /** @private */
    _getNextCorrId() {
        return ++this.corrIdCounter;
    }

    /**
     * Send a command and wait for a Response.
     * @private
     * @returns {Promise<{ok: boolean, result: string}>}
     */
    _sendMsgAndWait(msg) {
        return new Promise((resolve, reject) => {
            if (!this.ws || this.ws.readyState !== this.Socket.OPEN) {
                return reject(new Error("WebSocket is not connected."));
            }
            const corr_id = this._getNextCorrId();
            msg.corr_id = corr_id;
            this.pendingRequests.set(corr_id, { resolve, reject });

            try {
                const encoded = msgpack.encode(msg);
                this.ws.send(encoded);
            } catch (error) {
                this.pendingRequests.delete(corr_id);
                reject(error);
            }
        });
    }

    /** @private */
    async _sendMsg(msg) {
        if (!this.ws || this.ws.readyState !== this.Socket.OPEN) {
            throw new Error("WebSocket is not connected.");
        }
        const encoded = msgpack.encode(msg);
        this.ws.send(encoded);
    }

    // =========================================================================
    // Queries
    // =========================================================================

    /**
     * Sends a generic query to the server.
     * @param {string} subject The query subject.
     * @param {string} record The query record.
     * @returns {Promise<{ok: boolean, result: string}>}
     */
    async query(subject, record) {
        const msg = { type: "query", subject, record };
        return await this._sendMsgAndWait(msg);
    }

    /**
     * Check if a program exists on the server.
     * The inferlet must be in name@version format (e.g., "text-completion@0.1.0").
     * @param {string} inferlet The inferlet name (e.g., "text-completion@0.1.0").
     * @returns {Promise<boolean>}
     */
    async checkProgram(inferlet) {
        const idx = inferlet.lastIndexOf('@');
        if (idx === -1) throw new Error("Version required: use 'name@version' format");
        const name = inferlet.substring(0, idx);
        const version = inferlet.substring(idx + 1);
        const msg = { type: "check_program", name, version };
        const { ok, result } = await this._sendMsgAndWait(msg);
        if (ok) return result === "true";
        throw new Error(`CheckProgram failed: ${result}`);
    }

    // =========================================================================
    // Program Upload
    // =========================================================================

    /**
     * Installs a program to the server in chunks.
     * @param {string} wasmPath Path to the WASM binary file (Node.js only).
     * @param {string} manifestPath Path to the manifest TOML file (Node.js only).
     */
    async installProgram(wasmPath, manifestPath, forceOverwrite = false) {
        const fs = await import('fs');
        const programBytes = fs.readFileSync(wasmPath);
        const manifest = fs.readFileSync(manifestPath, 'utf-8');
        return await this.installProgramBytes(programBytes, manifest, forceOverwrite);
    }

    /**
     * Installs a program from its artifact bytes (a component, or a script's
     * source when the manifest names a `[runtime] language`) and its
     * manifest TOML.
     * @param {Uint8Array} programBytes
     * @param {string} manifest
     */
    async installProgramBytes(programBytes, manifest, forceOverwrite = false) {
        // msgpack-lite encodes a Node Buffer as `bin`, which is what the
        // server's `chunk_data` is; a bare Uint8Array would go out as an
        // array of numbers and the upload would never be answered.
        if (typeof Buffer !== 'undefined' && !Buffer.isBuffer(programBytes)) {
            programBytes = Buffer.from(programBytes.buffer, programBytes.byteOffset, programBytes.byteLength);
        }
        const programHash = hashBytes(programBytes);

        const totalChunks = Math.max(1, Math.ceil(programBytes.length / CHUNK_SIZE));
        const corr_id = this._getNextCorrId();

        const installPromise = new Promise((resolve, reject) => {
            this.pendingRequests.set(corr_id, { resolve, reject });
        });

        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, programBytes.length);
            const msg = {
                type: "add_program",
                corr_id,
                program_hash: programHash,
                manifest,
                force_overwrite: forceOverwrite,
                chunk_index: i,
                total_chunks: totalChunks,
                chunk_data: programBytes.slice(start, end),
            };
            await this._sendMsg(msg);
        }

        const { ok, result } = await installPromise;
        if (!ok) {
            throw new Error(`Program install failed: ${result}`);
        }
    }

    // =========================================================================
    // File Transfer (fire-and-forget)
    // =========================================================================

    /** @private */
    async _transferFile(processId, fileBytes) {
        const fileHash = hashBytes(fileBytes);
        const totalChunks = Math.max(1, Math.ceil(fileBytes.length / CHUNK_SIZE));

        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, fileBytes.length);
            const msg = {
                type: "transfer_file",
                process_id: processId,
                file_hash: fileHash,
                chunk_index: i,
                total_chunks: totalChunks,
                chunk_data: fileBytes.slice(start, end),
            };
            await this._sendMsg(msg);
        }
    }

    // =========================================================================
    // Process Lifecycle
    // =========================================================================

    /**
     * Launches a process. Returns a Process object for interaction.
     * @param {string} inferlet The inferlet name (e.g., "text-completion@0.1.0").
     * @param {Object} [input={}] Input parameters object, serialized to JSON.
     * @param {boolean} [captureOutputs=true] Stream outputs to client.
     * @returns {Promise<Process>}
     */
    async launchProcess(inferlet, input = {}, captureOutputs = true) {
        const msg = {
            type: "launch_process",
            inferlet,
            input: JSON.stringify(input),
            capture_outputs: captureOutputs,
        };
        const { ok, result } = await this._sendMsgAndWait(msg);
        if (!ok) {
            throw new Error(`Failed to launch process: ${result}`);
        }

        const processId = result;
        const queue = new AsyncQueue();
        this.processEventQueues.set(processId, queue);
        // Replay orphan events
        if (this.orphanEvents.has(processId)) {
            for (const tuple of this.orphanEvents.get(processId)) {
                queue.put(tuple);
            }
            this.orphanEvents.delete(processId);
        }

        return new Process(this, processId);
    }

    /**
     * Launches an inferlet with `input` as its input object. A function
     * readied by `inferlet()` is installed first if the server does not
     * hold its exact source yet; a string is an installed program's
     * `name@version`. Returns the `Process`; `await process.result()`
     * waits for what it returns.
     * @param {Inferlet|string} program
     * @param {object} input
     * @returns {Promise<Process>}
     */
    async run(program, input = {}) {
        if (program instanceof Inferlet) {
            if (!(await this.checkProgram(program.program))) {
                await this.installProgramBytes(
                    new TextEncoder().encode(program.source), program.manifestToml(),
                );
            }
            program = program.program;
        }
        return await this.launchProcess(program, input);
    }

    /**
     * Attaches to an existing process.
     * @param {string} processId The UUID of the process.
     * @returns {Promise<Process>}
     */
    async attachProcess(processId) {
        const msg = {
            type: "attach_process",
            process_id: processId,
        };
        const { ok, result } = await this._sendMsgAndWait(msg);
        if (!ok) {
            throw new Error(`Failed to attach to process: ${result}`);
        }

        const queue = new AsyncQueue();
        this.processEventQueues.set(processId, queue);
        if (this.orphanEvents.has(processId)) {
            for (const tuple of this.orphanEvents.get(processId)) {
                queue.put(tuple);
            }
            this.orphanEvents.delete(processId);
        }

        return new Process(this, processId);
    }

    /**
     * Sends a signal/message to a running process (fire-and-forget).
     * @param {string} processId The process UUID.
     * @param {string} message The message to send.
     */
    async signalProcess(processId, message) {
        const msg = { type: "signal_process", process_id: processId, message };
        await this._sendMsg(msg);
    }

    /**
     * Terminates a running process.
     * @param {string} processId The process UUID.
     */
    async terminateProcess(processId) {
        const msg = { type: "terminate_process", process_id: processId };
        const { ok, result } = await this._sendMsgAndWait(msg);
        if (!ok) {
            throw new Error(`Failed to terminate process: ${result}`);
        }
    }

    /**
     * Lists running processes.
     * @returns {Promise<string[]>} List of process UUID strings.
     */
    async listProcesses() {
        const msg = { type: "list_processes" };
        const { ok, result } = await this._sendMsgAndWait(msg);
        if (!ok) {
            throw new Error(`List processes failed: ${result}`);
        }
        try {
            return JSON.parse(result);
        } catch {
            return result ? [result] : [];
        }
    }

    /**
     * Pings the server.
     */
    async ping() {
        const msg = { type: "ping" };
        const { ok, result } = await this._sendMsgAndWait(msg);
        if (!ok) {
            throw new Error(`Ping failed: ${result}`);
        }
    }
}
