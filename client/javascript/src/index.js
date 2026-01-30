/**
 * @file pie-client.js
 * A JavaScript client library for the Pie WebSocket server.
 *
 * @requires msgpack-lite
 * @requires blake3
 */

// If using in a browser with script tags, these would be global.
// If using in Node.js or with a bundler, import them.
import msgpack from 'msgpack-lite';
import { blake3 } from 'blake3';

/**
 * A simple asynchronous queue.
 * Producers can push values, and consumers can await until a value is available.
 */
class AsyncQueue {
    constructor() {
        this._values = [];
        this._resolvers = [];
    }

    /**
     * Puts a value into the queue. If a consumer is waiting, it resolves their promise.
     * @param {*} value The value to add to the queue.
     */
    put(value) {
        if (this._resolvers.length > 0) {
            const resolve = this._resolvers.shift();
            resolve(value);
        } else {
            this._values.push(value);
        }
    }

    /**
     * Gets a value from the queue. If the queue is empty, it waits until a value is added.
     * @returns {Promise<*>} A promise that resolves with the next value in the queue.
     */
    get() {
        return new Promise((resolve) => {
            if (this._values.length > 0) {
                resolve(this._values.shift());
            } else {
                this._resolvers.push(resolve);
            }
        });
    }

    /**
     * Checks if the queue is empty.
     * @returns {boolean}
     */
    isEmpty() {
        return this._values.length === 0;
    }
}

/**
 * Represents a running instance of a program on the server.
 */
export class Instance {
    /**
     * @param {PieClient} client The PieClient that owns this instance.
     * @param {string} instanceId The unique ID of the instance.
     */
    constructor(client, instanceId) {
        this.client = client;
        this.instanceId = instanceId;
        this.eventQueue = client.instEventQueues.get(instanceId);
        if (!this.eventQueue) {
            throw new Error(`Internal error: No event queue for instance ${instanceId}`);
        }
    }

    /**
     * Sends a message to the instance.
     * @param {string} message The message to send.
     * @returns {Promise<void>}
     */
    async send(message) {
        await this.client.signalInstance(this.instanceId, message);
    }

    /**
     * Receives an event from the instance. Blocks until an event is available.
     * @returns {Promise<{event: string, msg: string}>} An object containing the event type and message.
     */
    async recv() {
        if (!this.eventQueue) {
            throw new Error("Event queue is not available for this instance.");
        }
        const [event, msg] = await this.eventQueue.get();
        return { event, msg };
    }

    /**
     * Requests termination of the instance on the server.
     * @returns {Promise<void>}
     */
    async terminate() {
        await this.client.terminateInstance(this.instanceId);
    }
}

/**
 * An asynchronous client for interacting with the Pie WebSocket server.
 */
export class PieClient {
    /**
     * @param {string} serverUri The WebSocket server URI (e.g., "ws://127.0.0.1:8080").
     */
    constructor(serverUri) {
        this.serverUri = serverUri;
        this.ws = null;
        this.corrIdCounter = 0;
        this.pendingRequests = new Map();
        this.instEventQueues = new Map();
        this.connectionPromise = null;
    }

    /**
     * Establishes a WebSocket connection and starts the background listener.
     * @returns {Promise<void>} A promise that resolves when the connection is open.
     */
    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }

        // If a connection is already in progress, return the existing promise
        if (this.connectionPromise) {
            return this.connectionPromise;
        }

        this.connectionPromise = new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.serverUri);
                this.ws.binaryType = 'blob'; // Use blob for easier conversion to ArrayBuffer

                this.ws.onopen = () => {
                    console.log(`[PieClient] Connected to ${this.serverUri}`);
                    this._listen();
                    resolve();
                };

                this.ws.onerror = (error) => {
                    console.error("[PieClient] WebSocket error:", error);
                    reject(new Error("WebSocket connection failed."));
                    this.connectionPromise = null; // Reset for future connection attempts
                };

                this.ws.onclose = (event) => {
                    console.log(`[PieClient] Connection closed. Code: ${event.code}, Reason: ${event.reason}`);
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

    /**
     * Background listener to process all incoming server messages.
     * @private
     */
    async _listen() {
        this.ws.onmessage = async (event) => {
            if (event.data instanceof Blob) {
                try {
                    const arrayBuffer = await event.data.arrayBuffer();
                    const message = msgpack.decode(new Uint8Array(arrayBuffer));
                    this._processServerMessage(message);
                } catch (e) {
                    console.error("[PieClient] Failed to decode messagepack:", e);
                }
            } else {
                console.log(`[PieClient] Received non-binary message: ${event.data}`);
            }
        };
    }

    /**
     * Routes incoming server messages based on their type.
     * @private
     * @param {object} message The decoded message object.
     */
    _processServerMessage(message) {
        const { type, corr_id, instance_id, event, message: msg, successful, result } = message;

        switch (type) {
            case 'response':
                if (this.pendingRequests.has(corr_id)) {
                    const promiseControls = this.pendingRequests.get(corr_id);
                    promiseControls.resolve({ successful, result });
                    this.pendingRequests.delete(corr_id);
                }
                break;
            case 'instance_event':
                if (this.instEventQueues.has(instance_id)) {
                    this.instEventQueues.get(instance_id).put([event, msg]);
                }
                break;
            case 'server_event':
                console.log(`[PieClient] Received server event: ${msg}`);
                break;
            default:
                console.warn(`[PieClient] Received unknown message type: ${type}`);
        }
    }

    /**
     * Gracefully closes the WebSocket connection.
     * @returns {Promise<void>}
     */
    close() {
        return new Promise((resolve) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.onclose = () => {
                    console.log("[PieClient] Client has been shut down.");
                    resolve();
                };
                this.ws.close();
            } else {
                resolve();
            }
        });
    }

    /**
     * Generates a unique correlation ID for a request.
     * @private
     */
    _getNextCorrId() {
        return ++this.corrIdCounter;
    }

    /**
     * Sends a message that expects a response and waits for it.
     * @private
     * @param {object} msg The message object to send.
     * @returns {Promise<{successful: boolean, result: string}>}
     */
    _sendMsgAndWait(msg) {
        return new Promise(async (resolve, reject) => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
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

    /**
     * Sends a fire-and-forget message.
     * @private
     * @param {object} msg The message object to send.
     */
    async _sendMsg(msg) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error("WebSocket is not connected.");
        }
        const encoded = msgpack.encode(msg);
        this.ws.send(encoded);
    }

    /**
     * Authenticates the client with the server using a token.
     * @param {string} token The JWT token.
     * @returns {Promise<{successful: boolean, result: string}>}
     */
    async authenticate(token) {
        const msg = { type: "authenticate", token };
        const { successful, result } = await this._sendMsgAndWait(msg);
        if (successful) {
            console.log("[PieClient] Authenticated successfully.");
        } else {
            console.error(`[PieClient] Authentication failed: ${result}`);
        }
        return { successful, result };
    }

    /**
     * Uploads a program to the server in chunks.
     * @param {Uint8Array} programBytes The program content as a byte array.
     * @param {string} manifest The manifest TOML content as a string.
     * @returns {Promise<void>}
     */
    async uploadProgram(programBytes, manifest) {
        const programHash = blake3(programBytes).toString('hex');
        const chunkSize = 256 * 1024; // 256 KiB, must match server
        const totalChunks = Math.ceil(programBytes.length / chunkSize);
        const corr_id = this._getNextCorrId();

        const uploadPromise = new Promise((resolve, reject) => {
            this.pendingRequests.set(corr_id, { resolve, reject });
        });

        for (let i = 0; i < totalChunks; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, programBytes.length);
            const chunkData = programBytes.slice(start, end);
            const msg = {
                type: "upload_program",
                corr_id: corr_id,
                program_hash: programHash,
                manifest: manifest,
                chunk_index: i,
                total_chunks: totalChunks,
                chunk_data: chunkData,
            };
            await this._sendMsg(msg);
        }

        const { successful, result } = await uploadPromise;
        if (successful) {
            console.log(`[PieClient] Program uploaded successfully: ${result}`);
        } else {
            throw new Error(`Program upload failed: ${result}`);
        }
    }

    /**
     * Launches an instance of a program.
     * @param {string} programHash The hash of the program to launch.
     * @param {string[]} [args=[]] Optional command-line arguments.
     * @returns {Promise<Instance>}
     */
    /**
     * Launches an instance of a program.
     *
     * This method performs a two-level search for the inferlet:
     * 1. First, it searches for the program among client-uploaded programs.
     * 2. If not found, it falls back to searching the registry.
     *
     * The inferlet parameter can be:
     * - Full name with version: "std/text-completion@0.1.0"
     * - Without namespace (defaults to "std"): "text-completion@0.1.0"
     * - Without version (defaults to "latest"): "std/text-completion" or "text-completion"
     *
     * @param {string} inferlet The inferlet name (e.g., "std/text-completion@0.1.0").
     * @param {string[]} [args=[]] Optional command-line arguments.
     * @param {boolean} [detached=false] If true, the instance runs in detached mode.
     * @returns {Promise<Instance>}
     */
    async launchInstance(inferlet, args = [], detached = false) {
        const msg = {
            type: "launch_instance",
            inferlet: inferlet,
            arguments: args,
            detached: detached,
        };
        const { successful, result } = await this._sendMsgAndWait(msg);
        if (successful) {
            const instanceId = result;
            this.instEventQueues.set(instanceId, new AsyncQueue());
            return new Instance(this, instanceId);
        } else {
            throw new Error(`Failed to launch instance: ${result}`);
        }
    }

    /**
     * Launches an instance from an inferlet in the registry only.
     *
     * Unlike `launchInstance`, this method searches only the registry and does not
     * check client-uploaded programs. Use this when you explicitly want to launch
     * an inferlet from the registry.
     * 
     * The inferlet parameter can be:
     * - Full name with version: "std/text-completion@0.1.0"
     * - Without namespace (defaults to "std"): "text-completion@0.1.0"
     * - Without version (defaults to "latest"): "std/text-completion" or "text-completion"
     * 
     * @param {string} inferlet The inferlet name (e.g., "std/text-completion@0.1.0").
     * @param {string[]} [args=[]] Optional command-line arguments.
     * @param {boolean} [detached=false] If true, the instance runs in detached mode.
     * @returns {Promise<Instance>}
     */
    async launchInstanceFromRegistry(inferlet, args = [], detached = false) {
        const msg = {
            type: "launch_instance_from_registry",
            inferlet: inferlet,
            arguments: args,
            detached: detached,
        };
        const { successful, result } = await this._sendMsgAndWait(msg);
        if (successful) {
            const instanceId = result;
            this.instEventQueues.set(instanceId, new AsyncQueue());
            return new Instance(this, instanceId);
        } else {
            throw new Error(`Failed to launch instance from registry: ${result}`);
        }
    }

    /**
     * Sends a signal/message to a running instance (fire-and-forget).
     * @param {string} instanceId The ID of the instance.
     * @param {string} message The message to send.
     * @returns {Promise<void>}
     */
    async signalInstance(instanceId, message) {
        const msg = { type: "signal_instance", instance_id: instanceId, message };
        await this._sendMsg(msg);
    }

    /**
     * Requests the server to terminate a running instance (fire-and-forget).
     * @param {string} instanceId The ID of the instance.
     * @returns {Promise<void>}
     */
    async terminateInstance(instanceId) {
        const msg = { type: "terminate_instance", instance_id: instanceId };
        await this._sendMsg(msg);
    }
}


// ===============================================================
// Example Usage:
// ===============================================================
// To run this example, you would do something like this in an async context:

async function main() {
    const client = new PieClient("ws://127.0.0.1:8080");

    try {
        await client.connect();

        // 1. Authenticate (if needed)
        // await client.authenticate("your-super-secret-jwt-token");

        // 2. Upload a simple program with manifest
        const programCode = new TextEncoder().encode('print("Hello from JavaScript instance!")');
        const manifest = `[package]
name = "example/hello-world"
version = "0.1.0"
`;
        await client.uploadProgram(programCode, manifest);
        console.log(`[Example] Uploaded program: example/hello-world@0.1.0`);

        // 3. Launch the instance using inferlet name
        console.log("[Example] Launching instance...");
        const instance = await client.launchInstance("example/hello-world@0.1.0");
        console.log(`[Example] Launched instance with ID: ${instance.instanceId}`);

        // 4. Wait for the instance to finish
        while (true) {
            const { event, msg } = await instance.recv();
            console.log(`[Example] Received event='${event}', message='${msg}'`);
            if (event === "terminated") {
                break;
            }
        }

    } catch (error) {
        console.error("[Example] An error occurred:", error);
    } finally {
        console.log("[Example] Closing client connection.");
        await client.close();
    }
}

// To run the main function:
// main();
