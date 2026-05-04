/**
 * @file mcp_bridge.js
 * Local bridge from this Node.js client to MCP servers.
 *
 * Mirrors the Rust/Python clients' bridges: spawns child processes for
 * `stdio` transport, multiplexes JSON-RPC requests over a single
 * connection, and performs the MCP `initialize` handshake at registration
 * time so the engine-side registration is only announced if the server
 * actually came up.
 *
 * Browser environments don't have `child_process`; importing this module
 * is fine, but `BridgeRegistry.registerStdio` will throw at call time.
 */

const PROTOCOL_VERSION = '2024-11-05';
const CLIENT_INFO = { name: 'pie-client-js', version: '1.0.0' };

/**
 * A JSON-RPC error returned by an MCP server (or synthesized for
 * transport-level failures).
 */
export class JsonRpcError extends Error {
    constructor(code, message, data = null) {
        super(`JSON-RPC error ${code}: ${message}`);
        this.code = code;
        this.data = data;
    }
}

/**
 * A connection to a single stdio MCP server. Many caller flows
 * multiplex over the same connection using distinct request ids.
 *
 * Construct via `StdioServer.spawn(...)` rather than `new`.
 */
export class StdioServer {
    constructor(name, child) {
        this.name = name;
        this.child = child;
        this._nextId = 1;
        this._pending = new Map();
        this._dead = false;
        this._stdoutBuf = '';

        child.stdout.setEncoding('utf8');
        child.stdout.on('data', (chunk) => this._onStdoutData(chunk));
        child.stdout.on('end', () => this._markDead());

        child.stderr.setEncoding('utf8');
        child.stderr.on('data', (chunk) => {
            for (const line of String(chunk).split('\n')) {
                if (line) console.error(`[mcp:${this.name}] ${line}`);
            }
        });

        child.on('exit', () => this._markDead());
        child.on('error', () => this._markDead());
    }

    /**
     * Spawn an MCP server process. Resolves once the child is alive;
     * the caller is expected to call `handshake()` next.
     */
    static async spawn(name, command, args = []) {
        const { spawn } = await import('node:child_process');
        const child = spawn(command, args, {
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        return new StdioServer(name, child);
    }

    async handshake() {
        await this.call('initialize', {
            protocolVersion: PROTOCOL_VERSION,
            capabilities: {},
            clientInfo: CLIENT_INFO,
        });
        // Notification — no response expected.
        this._writeJson({
            jsonrpc: '2.0',
            method: 'notifications/initialized',
            params: {},
        });
    }

    /**
     * Send a JSON-RPC request and await the matching response.
     * Resolves with the `result` field on success; rejects with
     * `JsonRpcError` on JSON-RPC `error` or transport failure.
     */
    call(method, params) {
        if (this._dead) {
            return Promise.reject(new JsonRpcError(
                -32000,
                `MCP server '${this.name}' is no longer running`,
            ));
        }
        return new Promise((resolve, reject) => {
            const id = this._nextId++;
            this._pending.set(id, { resolve, reject });
            try {
                this._writeJson({ jsonrpc: '2.0', id, method, params });
            } catch (err) {
                this._pending.delete(id);
                reject(new JsonRpcError(
                    -32000,
                    `MCP server '${this.name}' input closed: ${err.message}`,
                ));
            }
        });
    }

    /** @private */
    _writeJson(obj) {
        this.child.stdin.write(JSON.stringify(obj) + '\n');
    }

    /** @private */
    _onStdoutData(chunk) {
        this._stdoutBuf += chunk;
        let nl;
        while ((nl = this._stdoutBuf.indexOf('\n')) !== -1) {
            const line = this._stdoutBuf.slice(0, nl).trim();
            this._stdoutBuf = this._stdoutBuf.slice(nl + 1);
            if (!line) continue;
            this._handleLine(line);
        }
    }

    /** @private */
    _handleLine(line) {
        let msg;
        try {
            msg = JSON.parse(line);
        } catch (e) {
            console.error(`[mcp:${this.name}] non-JSON line (${e.message}): ${line}`);
            return;
        }
        const id = msg.id;
        if (typeof id !== 'number') return; // ignore notifications/server-initiated
        const slot = this._pending.get(id);
        if (!slot) return;
        this._pending.delete(id);
        if (msg.error) {
            const { code = -32000, message = 'MCP error', data = null } = msg.error;
            slot.reject(new JsonRpcError(code, message, data));
        } else {
            slot.resolve(msg.result ?? null);
        }
    }

    /** @private */
    _markDead() {
        if (this._dead) return;
        this._dead = true;
        for (const slot of this._pending.values()) {
            slot.reject(new JsonRpcError(
                -32000,
                `MCP server '${this.name}' died before response`,
            ));
        }
        this._pending.clear();
    }

    /**
     * Graceful shutdown ladder: close stdin (most MCP servers exit on
     * EOF) → SIGTERM → SIGKILL. Each step gives the server a brief
     * window to exit before escalating.
     */
    async close() {
        if (this.child.exitCode !== null || this.child.signalCode !== null) return;
        try { this.child.stdin.end(); } catch {}
        if (await this._waitForExit(1000)) return;
        try { this.child.kill('SIGTERM'); } catch {}
        if (await this._waitForExit(1000)) return;
        try { this.child.kill('SIGKILL'); } catch {}
        await this._waitForExit(1000);
    }

    /** @private */
    _waitForExit(ms) {
        return new Promise((resolve) => {
            if (this.child.exitCode !== null || this.child.signalCode !== null) {
                resolve(true);
                return;
            }
            const onExit = () => { clearTimeout(t); resolve(true); };
            const t = setTimeout(() => {
                this.child.removeListener('exit', onExit);
                resolve(false);
            }, ms);
            this.child.once('exit', onExit);
        });
    }
}

/**
 * Per-`PieClient` registry of locally-spawned MCP servers.
 */
export class BridgeRegistry {
    constructor() {
        this._servers = new Map();
    }

    /**
     * Spawn a stdio MCP server, complete the `initialize` handshake,
     * and publish it under `name`. Throws if `name` is already
     * registered or the handshake fails.
     */
    async registerStdio(name, command, args = []) {
        if (this._servers.has(name)) {
            throw new Error(`MCP server '${name}' already registered`);
        }
        const server = await StdioServer.spawn(name, command, args);
        try {
            await server.handshake();
        } catch (err) {
            await server.close();
            throw err;
        }
        this._servers.set(name, server);
    }

    get(name) {
        return this._servers.get(name) ?? null;
    }

    async closeAll() {
        const all = Array.from(this._servers.values());
        this._servers.clear();
        await Promise.all(all.map((s) => s.close()));
    }
}
